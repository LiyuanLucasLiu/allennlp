import json
import logging
from typing import Union, List, Dict, Any
import warnings

import torch
from torch.nn.modules import Dropout

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.common.util import lazy_groups_of
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn.util import remove_sentence_boundaries, add_sentence_boundary_token_ids, get_device_of
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.data.dataset import Batch
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# pylint: disable=attribute-defined-outside-init


class Elmo(torch.nn.Module):
    """
    Compute ELMo representations using a pre-trained bidirectional language model.

    See "Deep contextualized word representations", Peters et al. for details.

    This module takes character id input and computes ``num_output_representations`` different layers
    of ELMo representations.  Typically ``num_output_representations`` is 1 or 2.  For example, in
    the case of the SRL model in the above paper, ``num_output_representations=1`` where ELMo was included at
    the input token representation layer.  In the case of the SQuAD model, ``num_output_representations=2``
    as ELMo was also included at the GRU output layer.

    In the implementation below, we learn separate scalar weights for each output layer,
    but only run the biLM once on each input sequence for efficiency.

    Parameters
    ----------
    options_file : ``str``, required.
        ELMo JSON options file
    weight_file : ``str``, required.
        ELMo hdf5 weight file
    num_output_representations: ``int``, required.
        The number of ELMo representation layers to output.
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    do_layer_norm : ``bool``, optional, (default=False).
        Should we apply layer normalization (passed to ``ScalarMix``)?
    dropout : ``float``, optional, (default = 0.5).
        The dropout to be applied to the ELMo representations.
    vocab_to_cache : ``List[str]``, optional, (default = 0.5).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, Elmo expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    module : ``torch.nn.Module``, optional, (default = None).
        If provided, then use this module instead of the pre-trained ELMo biLM.
        If using this option, then pass ``None`` for both ``options_file``
        and ``weight_file``.  The module must provide a public attribute
        ``num_layers`` with the number of internal layers and its ``forward``
        method must return a ``dict`` with ``activations`` and ``mask`` keys
        (see `_ElmoBilm`` for an example).  Note that ``requires_grad`` is also
        ignored with this option.
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 num_output_representations: int,
                 requires_grad: bool = False,
                 do_layer_norm: bool = False,
                 dropout: float = 0.5,
                 vocab_to_cache: List[str] = None,
                 module: torch.nn.Module = None) -> None:
        super(Elmo, self).__init__()

        logging.info("Initializing ELMo")
        if module is not None:
            if options_file is not None or weight_file is not None:
                raise ConfigurationError(
                        "Don't provide options_file or weight_file with module")
            self._elmo_lstm = module
        else:
            self._elmo_lstm = _ElmoBiLm(options_file,
                                        weight_file,
                                        requires_grad=requires_grad,
                                        vocab_to_cache=vocab_to_cache)
        self._has_cached_vocab = vocab_to_cache is not None
        self._dropout = Dropout(p=dropout)
        self._scalar_mixes: Any = []
        for k in range(num_output_representations):
            scalar_mix = ScalarMix(self._elmo_lstm.num_layers, do_layer_norm=do_layer_norm)
            self.add_module('scalar_mix_{}'.format(k), scalar_mix)
            self._scalar_mixes.append(scalar_mix)

    def get_output_dim(self):
        return self._elmo_lstm.get_output_dim()

    def forward(self,    # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
        Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, required.
            If you passed a cached vocab, you can in addition pass a tensor of shape
            ``(batch_size, timesteps)``, which represent word ids which have been pre-cached.

        Returns
        -------
        Dict with keys:
        ``'elmo_representations'``: ``List[torch.Tensor]``
            A ``num_output_representations`` list of ELMo representations for the input sequence.
            Each representation is shape ``(batch_size, timesteps, embedding_dim)``
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, timesteps)`` long tensor with sequence mask.
        """
        # reshape the input if needed
        original_shape = inputs.size()
        if len(original_shape) > 3:
            timesteps, num_characters = original_shape[-2:]
            reshaped_inputs = inputs.view(-1, timesteps, num_characters)
        else:
            reshaped_inputs = inputs

        if word_inputs is not None:
            original_word_size = word_inputs.size()
            if self._has_cached_vocab and len(original_word_size) > 2:
                reshaped_word_inputs = word_inputs.view(-1, original_word_size[-1])
            elif not self._has_cached_vocab:
                logger.warning("Word inputs were passed to ELMo but it does not have a cached vocab.")
                reshaped_word_inputs = None
            else:
                reshaped_word_inputs = word_inputs
        else:
            reshaped_word_inputs = word_inputs

        # run the biLM
        bilm_output = self._elmo_lstm(reshaped_inputs, reshaped_word_inputs)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        # compute the elmo representations
        representations = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
            representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
            representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(
                    representation_with_bos_eos, mask_with_bos_eos
            )
            representations.append(self._dropout(representation_without_bos_eos))

        # reshape if necessary
        if word_inputs is not None and len(original_word_size) > 2:
            mask = mask_without_bos_eos.view(original_word_size)
            elmo_representations = [representation.view(original_word_size + (-1, ))
                                    for representation in representations]
        elif len(original_shape) > 3:
            mask = mask_without_bos_eos.view(original_shape[:-1])
            elmo_representations = [representation.view(original_shape[:-1] + (-1, ))
                                    for representation in representations]
        else:
            mask = mask_without_bos_eos
            elmo_representations = representations

        return {'elmo_representations': elmo_representations, 'mask': mask}

    # The add_to_archive logic here requires a custom from_params.
    @classmethod
    def from_params(cls, params: Params) -> 'Elmo':
        # Add files to archive
        params.add_file_to_archive('options_file')
        params.add_file_to_archive('weight_file')

        options_file = params.pop('options_file')
        weight_file = params.pop('weight_file')
        requires_grad = params.pop('requires_grad', False)
        num_output_representations = params.pop('num_output_representations')
        do_layer_norm = params.pop_bool('do_layer_norm', False)
        dropout = params.pop_float('dropout', 0.5)
        params.assert_empty(cls.__name__)

        return cls(options_file=options_file,
                   weight_file=weight_file,
                   num_output_representations=num_output_representations,
                   requires_grad=requires_grad,
                   do_layer_norm=do_layer_norm,
                   dropout=dropout)


def batch_to_ids(batch: List[List[str]]) -> torch.Tensor:
    """
    Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters
    (len(batch), max sentence length, max word length).

    Parameters
    ----------
    batch : ``List[List[str]]``, required
        A list of tokenized sentences.

    Returns
    -------
        A tensor of padded character ids.
    """
    instances = []
    indexer = ELMoTokenCharactersIndexer()
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens,
                          {'character_ids': indexer})
        instance = Instance({"elmo": field})
        instances.append(instance)

    dataset = Batch(instances)
    vocab = Vocabulary()
    dataset.index_instances(vocab)
    return dataset.as_tensor_dict()['elmo']['character_ids']


class _ElmoCharacterEncoder(torch.nn.Module):
    """
    Compute context insensitive token representation using pretrained biLM.

    This embedder has input character ids of size (batch_size, sequence_length, 50)
    and returns (batch_size, sequence_length + 2, embedding_dim), where embedding_dim
    is specified in the options file (typically 512).

    We add special entries at the beginning and end of each sequence corresponding
    to <S> and </S>, the beginning and end of sentence tokens.

    Note: this is a lower level class useful for advanced usage.  Most users should
    use ``ElmoTokenEmbedder`` or ``allennlp.modules.Elmo`` instead.

    Parameters
    ----------
    options_file : ``str``
        ELMo JSON options file
    weight_file : ``str``
        ELMo hdf5 weight file
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.

    The relevant section of the options file is something like:
    .. example-code::

        .. code-block:: python

            {'char_cnn': {
                'activation': 'relu',
                'embedding': {'dim': 4},
                'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                'max_characters_per_token': 50,
                'n_characters': 262,
                'n_highway': 2
                }
            }
    """
    def __init__(self,
                 options_file: str,
                 requires_grad: bool = False) -> None:
        super(_ElmoCharacterEncoder, self).__init__()

        with open(cached_path(options_file), 'r') as fin:
            self._options = json.load(fin)

        self.output_dim = self._options['lstm']['projection_dim']

        self._build_graph()

        self.requires_grad = requires_grad
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        # Cache the arrays for use in forward -- +1 due to masking.
        self._beginning_of_sentence_characters = torch.from_numpy(
                np.array(ELMoCharacterMapper.beginning_of_sentence_characters) + 1
        )
        self._end_of_sentence_characters = torch.from_numpy(
                np.array(ELMoCharacterMapper.end_of_sentence_characters) + 1
        )

    def get_output_dim(self):
        return self.output_dim

    @overrides
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        """
        Compute context insensitive token embeddings for ELMo representations.

        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, sequence_length, 50)`` of character ids representing the
            current batch.

        Returns
        -------
        Dict with keys:
        ``'token_embedding'``: ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 2, embedding_dim)`` tensor with context
            insensitive token representations.
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 2)`` long tensor with sequence mask.
        """
        # Add BOS/EOS
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()
        character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids(
                inputs,
                mask,
                self._beginning_of_sentence_characters,
                self._end_of_sentence_characters
        )

        # the character id embedding
        max_chars_per_token = self._options['char_cnn']['max_characters_per_token']
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(
                character_ids_with_bos_eos.view(-1, max_chars_per_token),
                self._char_embedding_weights
        )

        # run convolutions
        cnn_options = self._options['char_cnn']
        if cnn_options['activation'] == 'tanh':
            activation = torch.nn.functional.tanh
        elif cnn_options['activation'] == 'relu':
            activation = torch.nn.functional.relu
        else:
            raise ConfigurationError("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, 'conv_{}'.format(i))
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()

        return {
                'mask': mask_with_bos_eos,
                'token_embedding': token_embedding.view(batch_size, sequence_length, -1)
        }


    def _build_graph(self):
        # char embedding
        weights = (np.random.rand(
                 self._options['char_cnn']['n_characters'],
                 self._options['char_cnn']['embedding']['dim']
        ) * 2 - 1.0).astype('float32')
        self._char_embedding_weights = torch.nn.Parameter(
                torch.FloatTensor(weights)
        )

        # convolutions
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        char_embed_dim = cnn_options['embedding']['dim']

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                    in_channels=char_embed_dim,
                    out_channels=num,
                    kernel_size=width,
                    bias=True
            )
            conv.weight.data.uniform_(-0.05, 0.05)
            conv.bias.data.fill_(0.0)
            convolutions.append(conv)
            self.add_module('conv_{}'.format(i), conv)

        self._convolutions = convolutions

        # the highway layers
        if cnn_options['projection']['location'] == 'after_cnn':
            highway_dim = cnn_options['projection']['dim']
        else:
            # highway dim is the number of CNN filters
            highway_dim = sum(f[1] for f in filters)
        n_highway = cnn_options['n_highway']
        self._highways = Highway(
            highway_dim, n_highway, activation=torch.nn.functional.relu)
        for highway in self._highways._layers:
            # highway is a Linear layer for each highway layer with fused
            # W and b weights
            highway.weight.data.normal_(mean=0.0,
                                        std=np.sqrt(1.0 / highway_dim))
            highway.bias[:highway_dim].data.fill_(0.0)
            highway.bias[highway_dim:].data.fill_(2.0)
    
        # the projection layer -- always n_filters -> projection dim
        n_filters = sum(f[1] for f in filters)
        projection_dim = cnn_options['projection']['dim']
        self._projection = torch.nn.Linear(n_filters, projection_dim, bias=True)
        self._projection.weight.data.normal_(mean=0.0,
                                             std=np.sqrt(1.0 / n_filters))
        self._projection.bias.data.fill_(0.0)

class _ElmoBiLm(torch.nn.Module):
    """
    Run a pre-trained bidirectional language model, outputing the activations at each
    layer for weighting together into an ELMo representation (with
    ``allennlp.modules.seq2seq_encoders.Elmo``).  This is a lower level class, useful
    for advanced uses, but most users should use ``allennlp.modules.seq2seq_encoders.Elmo``
    directly.

    Parameters
    ----------
    options_file : ``str``
        ELMo JSON options file
    weight_file : ``str``
        ELMo hdf5 weight file
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    vocab_to_cache : ``List[str]``, optional, (default = 0.5).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, _ElmoBiLm expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 requires_grad: bool = False,
                 vocab_to_cache: List[str] = None) -> None:
        super(_ElmoBiLm, self).__init__()

        self._character_encoder = _ElmoCharacterEncoder(options_file, requires_grad=requires_grad)

        self._requires_grad = requires_grad
        if requires_grad and vocab_to_cache:
            logging.warning("You are fine tuning ELMo and caching char CNN word vectors. "
                            "This behaviour is not guaranteed to be well defined, particularly. "
                            "if not all of your inputs will occur in the vocabulary cache.")
        # This is an embedding, used to look up cached
        # word vectors built from character level cnn embeddings.
        self._word_embedding = None
        self._bos_embedding: torch.Tensor = None
        self._eos_embedding: torch.Tensor = None
        if vocab_to_cache:
            logging.info("Caching character cnn layers for words in vocabulary.")
            # This sets 3 attributes, _word_embedding, _bos_embedding and _eos_embedding.
            # They are set in the method so they can be accessed from outside the
            # constructor.
            self.create_cached_cnn_embeddings(vocab_to_cache)

        with open(cached_path(options_file), 'r') as fin:
            options = json.load(fin)
        if not options['lstm'].get('use_skip_connections'):
            raise ConfigurationError('We only support pretrained biLMs with residual connections')
        self._elmo_lstm = ElmoLstm(input_size=options['lstm']['projection_dim'],
                                   hidden_size=options['lstm']['projection_dim'],
                                   cell_size=options['lstm']['dim'],
                                   num_layers=options['lstm']['n_layers'],
                                   two_head=options['lstm'].get('two_head', False),
                                   memory_cell_clip_value=options['lstm']['cell_clip'],
                                   state_projection_clip_value=options['lstm']['proj_clip'],
                                   requires_grad=requires_grad)
        # self._elmo_lstm.load_weights(weight_file)

        weight_file = torch.load(cached_path(weight_file), map_location=lambda storage, loc: storage)

        self._character_encoder.load_state_dict(weight_file['_character_encoder'])
        self._elmo_lstm.load_state_dict(weight_file['_elmo_lstm'])
        
        # Number of representation layers including context independent layer
        self.num_layers = options['lstm']['n_layers'] + 1

    def get_output_dim(self):
        return 2 * self._character_encoder.get_output_dim()

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, required.
            If you passed a cached vocab, you can in addition pass a tensor of shape ``(batch_size, timesteps)``,
            which represent word ids which have been pre-cached.

        Returns
        -------
        Dict with keys:

        ``'activations'``: ``List[torch.Tensor]``
            A list of activations at each layer of the network, each of shape
            ``(batch_size, timesteps + 2, embedding_dim)``
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, timesteps + 2)`` long tensor with sequence mask.

        Note that the output tensors all include additional special begin and end of sequence
        markers.
        """
        if self._word_embedding is not None and word_inputs is not None:
            try:
                mask_without_bos_eos = (word_inputs > 0).long()
                # The character cnn part is cached - just look it up.
                embedded_inputs = self._word_embedding(word_inputs) # type: ignore
                # shape (batch_size, timesteps + 2, embedding_dim)
                type_representation, mask = add_sentence_boundary_token_ids(
                        embedded_inputs,
                        mask_without_bos_eos,
                        self._bos_embedding,
                        self._eos_embedding
                )
            except RuntimeError:
                # Back off to running the character convolutions,
                # as we might not have the words in the cache.
                token_embedding = self._character_encoder(inputs)
                mask = token_embedding['mask']
                type_representation = token_embedding['token_embedding']
        else:
            token_embedding = self._character_encoder(inputs)
            mask = token_embedding['mask']
            type_representation = token_embedding['token_embedding']
        lstm_outputs = self._elmo_lstm(type_representation, mask)

        # Prepare the output.  The first layer is duplicated.
        # Because of minor differences in how masking is applied depending
        # on whether the char cnn layers are cached, we'll be defensive and
        # multiply by the mask here. It's not strictly necessary, as the
        # mask passed on is correct, but the values in the padded areas
        # of the char cnn representations can change.
        output_tensors = [
                torch.cat([type_representation, type_representation], dim=-1) * mask.float().unsqueeze(-1)
        ]
        for layer_activations in torch.chunk(lstm_outputs, lstm_outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))

        return {
                'activations': output_tensors,
                'mask': mask,
        }

    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None:
        """
        Given a list of tokens, this method precomputes word representations
        by running just the character convolutions and highway layers of elmo,
        essentially creating uncontextual word vectors. On subsequent forward passes,
        the word ids are looked up from an embedding, rather than being computed on
        the fly via the CNN encoder.

        This function sets 3 attributes:

        _word_embedding : ``torch.Tensor``
            The word embedding for each word in the tokens passed to this method.
        _bos_embedding : ``torch.Tensor``
            The embedding for the BOS token.
        _eos_embedding : ``torch.Tensor``
            The embedding for the EOS token.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            A list of tokens to precompute character convolutions for.
        """
        tokens = [ELMoCharacterMapper.bos_token, ELMoCharacterMapper.eos_token] + tokens
        timesteps = 32
        batch_size = 32
        chunked_tokens = lazy_groups_of(iter(tokens), timesteps)

        all_embeddings = []
        device = get_device_of(next(self.parameters()))
        for batch in lazy_groups_of(chunked_tokens, batch_size):
            # Shape (batch_size, timesteps, 50)
            batched_tensor = batch_to_ids(batch)
            # NOTE: This device check is for when a user calls this method having
            # already placed the model on a device. If this is called in the
            # constructor, it will probably happen on the CPU. This isn't too bad,
            # because it's only a few convolutions and will likely be very fast.
            if device >= 0:
                batched_tensor = batched_tensor.cuda(device)
            output = self._character_encoder(batched_tensor)
            token_embedding = output["token_embedding"]
            mask = output["mask"]
            token_embedding, _ = remove_sentence_boundaries(token_embedding, mask)
            all_embeddings.append(token_embedding.view(-1, token_embedding.size(-1)))
        full_embedding = torch.cat(all_embeddings, 0)

        # We might have some trailing embeddings from padding in the batch, so
        # we clip the embedding and lookup to the right size.
        full_embedding = full_embedding[:len(tokens), :]
        embedding = full_embedding[2:len(tokens), :]
        vocab_size, embedding_dim = list(embedding.size())

        from allennlp.modules.token_embedders import Embedding # type: ignore
        self._bos_embedding = full_embedding[0, :]
        self._eos_embedding = full_embedding[1, :]
        self._word_embedding = Embedding(vocab_size, # type: ignore
                                         embedding_dim,
                                         weight=embedding.data,
                                         trainable=self._requires_grad,
                                         padding_index=0)
