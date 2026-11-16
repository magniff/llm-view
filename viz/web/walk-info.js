'use strict';
// Longer explanations for the Step-by-step tab: one per step, in prose, with
// further reading. Shown when the step's info button is on. **bold**,
// `code` and $TeX$ are the only markup. Every link was checked to resolve to the page or
// video named here.

const WI_WIKI = (page) => 'https://en.wikipedia.org/wiki/' + page;
const WI_ARXIV = (id) => 'https://arxiv.org/abs/' + id;
const WI_YT = (id) => 'https://www.youtube.com/watch?v=' + id;

const L_ = {
  aiayn: ['paper', 'Vaswani et al. 2017, Attention Is All You Need (the original transformer)', WI_ARXIV('1706.03762')],
  illustrated: ['article', 'Jay Alammar, The Illustrated Transformer', 'https://jalammar.github.io/illustrated-transformer/'],
  illustratedGpt2: ['article', 'Jay Alammar, The Illustrated GPT-2 (decoder-only models, masking)', 'https://jalammar.github.io/illustrated-gpt2/'],
  yt3b1b5: ['video', '3Blue1Brown, Transformers, the tech behind LLMs (Deep Learning ch. 5)', WI_YT('wjZofJX0v4M')],
  yt3b1b6: ['video', '3Blue1Brown, Attention in transformers, step-by-step (ch. 6)', WI_YT('eMlx5fFNoYc')],
  yt3b1b7: ['video', '3Blue1Brown, How might LLMs store facts (ch. 7, the MLP)', WI_YT('9-Jl0dxWQs8')],
  ytGpt: ['video', 'Andrej Karpathy, Let\'s build GPT: from scratch, in code, spelled out', WI_YT('kCc8FmEb1nY')],
  ytTok: ['video', 'Andrej Karpathy, Let\'s build the GPT Tokenizer', WI_YT('zduSFxRajkE')],
  ytDeep: ['video', 'Andrej Karpathy, Deep Dive into LLMs like ChatGPT', WI_YT('7xTGNNLPyMI')],
  circuits: ['article', 'Elhage et al., A Mathematical Framework for Transformer Circuits', 'https://transformer-circuits.pub/2021/framework/index.html'],
  rmsnorm: ['paper', 'Zhang & Sennrich 2019, Root Mean Square Layer Normalization', WI_ARXIV('1910.07467')],
  layernorm: ['paper', 'Ba et al. 2016, Layer Normalization', WI_ARXIV('1607.06450')],
  prenorm: ['paper', 'Xiong et al. 2020, On Layer Normalization in the Transformer Architecture (pre-norm vs post-norm)', WI_ARXIV('2002.04745')],
  resnet: ['paper', 'He et al. 2015, Deep Residual Learning for Image Recognition', WI_ARXIV('1512.03385')],
  rope: ['paper', 'Su et al. 2021, RoFormer: Enhanced Transformer with Rotary Position Embedding', WI_ARXIV('2104.09864')],
  ropeBlog: ['article', 'EleutherAI, Rotary Embeddings: A Relative Revolution', 'https://blog.eleuther.ai/rotary-embeddings/'],
  gqa: ['paper', 'Ainslie et al. 2023, GQA: grouped-query attention', WI_ARXIV('2305.13245')],
  mqa: ['paper', 'Shazeer 2019, Fast Transformer Decoding: One Write-Head is All You Need (multi-query attention)', WI_ARXIV('1911.02150')],
  glu: ['paper', 'Shazeer 2020, GLU Variants Improve Transformer (SwiGLU)', WI_ARXIV('2002.05202')],
  gcnn: ['paper', 'Dauphin et al. 2016, Language Modeling with Gated Convolutional Networks (the original GLU)', WI_ARXIV('1612.08083')],
  swish: ['paper', 'Ramachandran et al. 2017, Searching for Activation Functions (Swish / SiLU)', WI_ARXIV('1710.05941')],
  kvmem: ['paper', 'Geva et al. 2020, Transformer Feed-Forward Layers Are Key-Value Memories', WI_ARXIV('2012.14913')],
  sinks: ['paper', 'Xiao et al. 2023, Efficient Streaming Language Models with Attention Sinks', WI_ARXIV('2309.17453')],
  massive: ['paper', 'Sun et al. 2024, Massive Activations in Large Language Models', WI_ARXIV('2402.17762')],
  tied: ['paper', 'Press & Wolf 2016, Using the Output Embedding to Improve Language Models (tied embeddings)', WI_ARXIV('1608.05859')],
  bpe: ['paper', 'Sennrich et al. 2015, Neural Machine Translation of Rare Words with Subword Units (BPE for NLP)', WI_ARXIV('1508.07909')],
  hfBpe: ['article', 'Hugging Face LLM course, Byte-Pair Encoding tokenization', 'https://huggingface.co/learn/llm-course/chapter6/5'],
  smol: ['paper', 'Allal et al. 2025, SmolLM2: When Smol Goes Big (this model)', WI_ARXIV('2502.02737')],
  card: ['article', 'SmolLM2-135M-Instruct model card', 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct'],
  w: (page, label) => ['wiki', label || decodeURIComponent(page).replace(/_/g, ' '), WI_WIKI(page)],
};

const WALK_INFO = {
  tok: {
    p: [
      'A neural network computes with numbers, so the first job is to turn text into a sequence of integers. SmolLM2 uses **byte-pair encoding (BPE)**: training started from the 256 possible bytes and repeatedly merged the most frequent adjacent pair in a large text corpus into a new vocabulary entry, until there were 49,152 entries. Frequent words end up as single tokens; rare words are spelled out of a few pieces.',
      'Because the base units are bytes, any text at all can be encoded: there is no "unknown word". Before merging, a pre-tokenizer splits the text into rough words, and the space in front of a word stays attached to it. That is why `" capital"` (with its space) and `"capital"` are different tokens with unrelated ids. This tokenizer also splits every digit into its own token, which is one reason the model does arithmetic digit by digit (and not very well).',
      'Everything downstream is counted in tokens: the context window (8,192 tokens for this model; the chat on this site keeps 2,048), the compute per step, the KV cache. The tokenizer is fixed before the model is trained and never changes afterwards.',
    ],
    links: [L_.w('Byte_pair_encoding'), L_.hfBpe, L_.ytTok, L_.bpe, L_.w('Lexical_analysis')],
  },
  onehot: {
    p: [
      'A token id is a category, not a quantity: token 3575 is not "more" than token 504. The standard way to feed a category into linear algebra is **one-hot encoding**: a vector as long as the vocabulary, all zeros except a single 1 at the id\'s position.',
      'Written this way, the next step is an ordinary matrix multiplication, and the embedding table is just the weight matrix of a linear layer with no bias. That view matters for training: the gradient that reaches the table only touches the rows of tokens that were actually used.',
      'The same idea appears at the other end. When the model is trained, the "correct answer" for each position is a one-hot vector over the vocabulary (the actual next token), and the loss compares the model\'s predicted distribution with it.',
    ],
    links: [L_.w('One-hot'), L_.w('Matrix_multiplication')],
  },
  embed: {
    p: [
      'Multiplying a one-hot row by the table selects exactly one row, so engines skip the multiplication and copy the row: an **embedding lookup**. The table is `[49152, 576]`, 28.3 million numbers, about 21% of the whole model. Its rows start random and are adjusted during training like every other weight.',
      'After training, tokens that are used in similar contexts end up with vectors pointing in similar directions (high cosine similarity). Individual coordinates usually have no meaning of their own: meaning lives in directions, and the model packs many more features than 576 into the space by using directions that are almost, but not quite, perpendicular.',
      'These rows are the start of the **residual stream**: one 576-number vector per token that every layer reads from and adds to. SmolLM2 reuses this same table at the very end to turn vectors back into token scores ("tied embeddings"), which saves 28 million parameters, a big deal for a small model.',
    ],
    links: [L_.w('Word_embedding'), L_.yt3b1b5, L_.illustrated, L_.tied],
  },
  stream: {
    p: [
      'Every layer has the form $x \\leftarrow x + \\operatorname{attn}(x)$ followed by $x \\leftarrow x + \\operatorname{mlp}(x)$. Nothing ever replaces the stream; layers only add to it. So the vector entering layer L is the embedding plus 2·L writes, and it can be read as a shared workspace that every layer reads from and writes into.',
      'The rows grow along the way, often by an order of magnitude, and a few coordinates become enormous ("massive activations"). That growth is one reason every block starts with a normalisation: each block should see inputs of a predictable size whatever happened before it.',
    ],
    links: [L_.circuits, L_.w('Residual_neural_network'), L_.massive],
  },
  rms: {
    p: [
      'Each attention and MLP block begins by normalising its input. SmolLM2 uses **RMSNorm**: compute the root mean square of the row, $\\sqrt{\\operatorname{mean}(x^2) + \\varepsilon}$, then divide by it and scale by a learned gain. ε = 1e-5 only prevents dividing by zero.',
      'The original transformer used **LayerNorm**, which also subtracts the mean and adds a learned bias. RMSNorm drops both; Zhang & Sennrich found it works as well while being cheaper, and Llama-style models adopted it.',
      'Where the norm sits matters too. The original transformer normalised *after* adding to the stream ("post-norm"). Modern models normalise the copy that goes *into* each block ("pre-norm") and leave the stream itself alone, which keeps a clean path through all 30 layers and makes deep models much easier to train.',
    ],
    links: [L_.rmsnorm, L_.layernorm, L_.prenorm, L_.w('Normalization_(machine_learning)'), L_.w('Root_mean_square')],
  },
  div: {
    p: [
      'Dividing by the RMS makes the block **scale-invariant**: multiply a row by 10 and the normalised result is identical. Every row now has RMS 1, so its length is √576 = 24 whatever it was before.',
      'This is what lets the stream grow freely across layers without destabilising the blocks that read it. A side effect: one huge coordinate (they exist, see "massive activations") dominates the RMS and shrinks every other coordinate of that row after the division.',
    ],
    links: [L_.rmsnorm, L_.massive],
  },
  gain: {
    p: [
      'The last part of RMSNorm is an element-wise multiplication by a learned vector `g` (here `input_layernorm.weight`, 576 numbers). It starts at all ones and training reshapes it. It is not a matrix multiply: dimension i of every token is scaled by the same g[i].',
      'Think of it as the block choosing how loudly to hear each dimension of the stream. Every norm has its own g: two per layer (before attention and before the MLP) and one final one, 61 in total.',
    ],
    links: [L_.rmsnorm, L_.w('Hadamard_product_(matrices)', 'Element-wise (Hadamard) product')],
  },
  q: {
    p: [
      'Attention needs three different views of every token, and each is a plain linear projection of the normalised row. The first is the **query**: $q = \\mathrm{xn}\\, W_Q^{\\top}$. Each of its 576 outputs is a dot product of the input row with one row of the weight matrix. There is no bias in Llama-family models.',
      'The ᵀ comes from a storage convention: PyTorch\'s `nn.Linear` keeps weights as `[out, in]` and computes $x\\,W^{\\top}$. The shape rule `[a, b] × [b, c] = [a, c]` tells you what comes out: the shared 576 disappears.',
      'Intuitively, a query encodes "what is this position looking for?" Which questions are useful is not designed by anyone; it is whatever made next-token prediction work during training.',
    ],
    links: [L_.yt3b1b6, L_.w('Matrix_multiplication'), L_.w('Transpose'), L_.aiayn],
  },
  k: {
    p: [
      'The **key** is a second projection of the same input: what a token offers to queries looking for something. Attention will compare every query with the keys of all tokens up to and including itself.',
      'A token\'s key depends only on that token and the ones before it, so it never changes once computed. That is why it can be stored and reused: the **KV cache** holds every layer\'s keys and values, and generating the next token only computes keys for the one new token.',
      'Here keys are 192 wide, a third of the queries\' 576. The "split into heads" step explains why.',
    ],
    links: [L_.yt3b1b6, L_.w('Attention_(machine_learning)'), L_.illustrated],
  },
  v: {
    p: [
      'The **value** is the third projection: the information a token hands over when another token attends to it. Queries and keys only decide *who listens to whom*; values are *what is said*.',
      'Per head, the value projection (576 → 64) and the later output projection (64 → 576) together form one low-rank map from a source token\'s stream to the destination token\'s stream. Interpretability work calls this the head\'s "OV circuit", and the query/key pair its "QK circuit".',
    ],
    links: [L_.yt3b1b6, L_.circuits, L_.illustrated],
  },
  heads: {
    p: [
      '**Multi-head attention** runs several small attentions in parallel instead of one big one. Each head gets its own 64-dimensional slice of q, k and v, computes its own attention pattern, and specialises: some heads look at the previous token, some at the same word earlier on, some at punctuation, and many at the very first token.',
      'In the original design every query head had its own key and value head. **Multi-query attention** (Shazeer 2019) shared one k/v head among all query heads to shrink the cache; **grouped-query attention** (Ainslie et al. 2023) is the middle ground used here: 9 query heads in 3 groups, each group sharing one k/v head.',
      'The saving is in the KV cache: per token, 30 layers × (k + v) × 192 numbers × 4 bytes = 45 KB here, against 135 KB with a k/v head for every query head. For long contexts the cache, not the weights, becomes the memory bottleneck.',
    ],
    links: [L_.gqa, L_.mqa, L_.aiayn, L_.yt3b1b6],
  },
  rope: {
    p: [
      'Attention by itself is blind to order: shuffle the tokens and every score stays the same. Position has to be injected somewhere. GPT-2 added a learned position vector to each embedding; the original transformer added fixed sine waves. Llama-style models use **rotary position embedding (RoPE)**, applied to queries and keys right before they are compared.',
      'Each head\'s 64 numbers form 32 pairs, and each pair is treated as a point in a plane and rotated by $m\\,\\theta_i$ ($m$ = position), with $\\theta_i = 100000^{-2i/64}$. Pair 0 turns a full radian per position; the last pair barely turns over hundreds of thousands. (This model pairs dimension i with i+32, the Hugging Face layout; the paper pairs neighbours. It is the same thing with the dimensions reordered.)',
      'The point of rotating is the dot product: rotating q by angle m and k by angle n gives a product that depends only on **m − n**. So attention sees relative distance, and "the previous token" means the same at position 5 and at position 5,000. Rotations also keep lengths, have no learned parameters, and are not applied to values.',
    ],
    links: [L_.rope, L_.ropeBlog, L_.w('Rotation_matrix'), L_.w('Euler%27s_formula', 'Euler\'s formula (the complex-number view of RoPE)')],
  },
  scores: {
    p: [
      'For each head, the score of row i against column j is the **dot product** of token i\'s query with token j\'s key: large when they point the same way. In matrix form it is $q\\,k^{\\top}$, a `[T, T]` grid per head, which is why attention costs grow with the square of the context length.',
      'The scores are divided by √64 = 8. The dot product of two random 64-dimensional vectors has a spread that grows like √64; without the division the scores would be large, softmax would put nearly all weight on one token, and training would get almost no gradient through the others.',
    ],
    links: [L_.w('Dot_product'), L_.aiayn, L_.yt3b1b6],
  },
  mask: {
    p: [
      'A language model must predict each token from the ones before it only. During training the whole text is processed at once, every position predicting its successor in parallel, so the model has to be stopped from simply looking at the answer. The **causal mask** sets every score where j > i to −∞, which softmax turns into exactly zero weight.',
      'This is what makes a "decoder-only" model like GPT or Llama. A useful consequence: what a token computes never depends on later tokens, so all of its keys and values stay valid as the text grows, which is exactly what the KV cache relies on.',
    ],
    links: [L_.illustratedGpt2, L_.ytGpt, L_.yt3b1b6],
  },
  softmax: {
    p: [
      '**Softmax** turns a row of scores into a probability distribution: exponentiate each, divide by the sum. Bigger scores get exponentially more weight, everything is positive, and each row sums to 1. Implementations subtract the row\'s maximum first so that the exponentials cannot overflow; the result is the same.',
      'You will often see heads put most of their weight on token 0 even when it is irrelevant. That is an **attention sink**: since the weights must sum to 1, a head with nothing useful to read parks its attention on a token whose value contributes almost nothing. It is learned behaviour, closely tied to the massive activations of the first token, and it is why streaming-generation tricks keep the first few tokens around forever.',
    ],
    links: [L_.w('Softmax_function'), L_.sinks, L_.massive],
  },
  mix: {
    p: [
      'Each head now builds, for every token, a **weighted average of the value vectors** of the tokens it may see, using the attention weights from softmax. In matrix form it is $P\\,v$: `[T, T] × [T, 64] → [T, 64]` per head.',
      'This is the only place in the entire network where information moves *between* positions. Everything else (norms, projections, the MLP) works on each row separately. The 9 heads\' 64-wide outputs are then laid side by side into one 576-wide row.',
    ],
    links: [L_.yt3b1b6, L_.illustrated],
  },
  o: {
    p: [
      'The **output projection** `o_proj` maps the concatenated head outputs back into the stream\'s 576 dimensions. Because concatenation followed by a matrix multiply is the same as each head multiplying by its own 64-column slice of the matrix and summing, you can equally say: every head writes its own vector into the stream, and the writes add up.',
      'That per-head view is what interpretability work uses to ask what an individual head contributes (the Flow tab, hidden for now, shows each head\'s write).',
    ],
    links: [L_.circuits, L_.aiayn],
  },
  add1: {
    p: [
      'Attention\'s output is **added** to the stream: a residual connection. Residual connections were introduced for very deep image networks (ResNet): with an identity path around every block, gradients flow straight through all the layers during training, and each block only has to learn a correction to what is already there.',
      'In a transformer this gives the stream its character: a running sum that every block reads and writes. Information written by layer 3 can still be read, unchanged, by layer 25.',
    ],
    links: [L_.resnet, L_.w('Residual_neural_network'), L_.circuits],
  },
  norm2: {
    p: [
      'The MLP block gets its own pre-norm: the same RMSNorm computation with its own gain vector. The checkpoint calls it `post_attention_layernorm`, which describes where it sits (after attention\'s residual add); its job is to be the MLP\'s input norm.',
    ],
    links: [L_.rmsnorm, L_.prenorm],
  },
  gateup: {
    p: [
      'The second half of every layer is the **MLP** (multilayer perceptron, also called the feed-forward network). It works on each token separately: it never looks at other positions. First it widens the row from 576 to 1,536 numbers, the "neurons", using two different matrices: `gate_proj` and `up_proj`.',
      'The original transformer widened 4× with a single matrix. Gated MLPs need two input matrices, so the width is cut to two thirds to keep the parameter count the same: 4 × 576 × 2/3 = 1,536 exactly. The MLPs hold about three quarters of each layer\'s weights (2.65 M of 3.54 M), and much of what a model "knows" seems to be stored here.',
    ],
    links: [L_.yt3b1b7, L_.w('Multilayer_perceptron'), L_.glu],
  },
  silu: {
    p: [
      'Without a nonlinearity, any stack of linear layers collapses into a single linear map, however deep. The **activation function** is what lets the network compute anything more interesting. SmolLM2 uses **SiLU** (also called Swish): $\\operatorname{silu}(x) = x\\,\\sigma(x)$.',
      'For large positive inputs it is nearly the identity; for large negative inputs nearly 0; in between it is smooth and dips slightly below zero (minimum about −0.28 near x = −1.28). The original transformer used ReLU ($\\max(0, x)$) and GPT-2 used GELU, a close relative of SiLU.',
    ],
    links: [L_.w('Swish_function'), L_.w('Activation_function'), L_.swish],
  },
  mul: {
    p: [
      'The activated gate multiplies the up projection element by element: $\\text{hidden} = \\operatorname{silu}(\\text{gate}) \\odot \\text{up}$. This is **SwiGLU**, a gated linear unit with a SiLU gate. The gate decides how much of each neuron gets through, and the up value supplies its size and sign.',
      'Gating comes from Dauphin et al. (GLU, 2016); Shazeer (2020) showed that GLU variants make transformers noticeably better for the same compute, and they have been standard since. One useful picture: each neuron\'s row of `gate_proj` detects a pattern in the input, and when it fires, the neuron writes its column of `down_proj` (next step) into the stream, like a key-value memory.',
    ],
    links: [L_.glu, L_.gcnn, L_.kvmem, L_.w('Gating_mechanism')],
  },
  down: {
    p: [
      'The **down projection** maps the 1,536 neurons back to 576 dimensions. Each neuron contributes its own column of `down_proj`, scaled by its activation, and the 1,536 contributions add up to the MLP\'s write. Since most activations are near zero, a few dozen neurons usually account for most of it.',
      'Geva et al. read the MLP as a **key-value memory**: the input side detects patterns, the output side writes associated content. 3Blue1Brown\'s chapter 7 walks through how a fact like "Michael Jordan plays basketball" could be stored this way.',
    ],
    links: [L_.yt3b1b7, L_.kvmem],
  },
  add2: {
    p: [
      'The MLP\'s write is added to the stream, and the layer is complete: same shape in, same shape out, `[T, 576]`. That uniform shape is what makes the architecture stackable: layer 1 cannot tell whether its input came from the embeddings or from 29 earlier layers.',
      'A consequence worth knowing: since the stream always lives in the same space, you can apply the final norm and the output projection to it after *any* layer and read off what the model "would predict now". That trick is called the logit lens.',
    ],
    links: [L_.resnet, L_.circuits],
  },
  repeat: {
    p: [
      'SmolLM2-135M has 30 identical layers, each with its own 9 weight tensors: about 3.54 million parameters per layer (0.88 M attention, 2.65 M MLP). With the 28.3 M embedding table that comes to 134.5 million, the "135M" in the name.',
      'The layers do not have assigned roles, but loose patterns show up in many models: early layers handle local, surface-level structure, middle layers assemble meaning and recall facts, and late layers turn that into a concrete prediction. Bigger models mostly add more and wider layers of the same kind.',
    ],
    links: [L_.smol, L_.card, L_.ytGpt, L_.ytDeep],
  },
  fnorm: {
    p: [
      'After the last layer, the stream gets one final RMSNorm with its own gain (`model.norm.weight`). The output projection that follows expects a normalised input, just as every block did.',
    ],
    links: [L_.rmsnorm],
  },
  logits: {
    p: [
      'To score every possible next token, each row is compared, by dot product, with every row of the embedding table: `[T, 576] × [576, 49152]`. Because SmolLM2 ties its embeddings, this is the very table used at the start. A token scores high when the final vector points in the same direction as that token\'s embedding.',
      'The scores are called **logits**: unnormalised log-probabilities. Only their differences matter; adding the same constant to all of them changes nothing. This one matrix multiply (28 million multiply-adds per token) is bigger than any single layer\'s work.',
    ],
    links: [L_.w('Logit'), L_.tied, L_.yt3b1b5],
  },
  probs: {
    p: [
      'Softmax over the 49,152 logits gives the next-token distribution. Every row predicts the token that follows it, so one pass over a prompt yields T predictions at once. Training uses exactly that: the loss is the **cross-entropy**, the average of −log(probability given to the actual next token), over every position of trillions of tokens of text.',
      'Generation uses only the last row. A sampler picks a token from it (temperature, top-p and the penalties in the controls above all act here, after the model), the token is appended, and the next forward pass only needs to process that one new token, because the keys and values of everything before it are already in the KV cache.',
    ],
    links: [L_.w('Softmax_function'), L_.w('Cross-entropy'), L_.ytDeep, L_.ytGpt],
  },
};

// **bold**, `code` and $maths$ → nodes.
function infoText(s) {
  return s.split(/(\*\*[^*]+\*\*|`[^`]+`|\$[^$]+\$)/).filter(Boolean).map((part) =>
    part.startsWith('**') ? el('b', { text: part.slice(2, -2) })
      : part.startsWith('`') ? el('code', { text: part.slice(1, -1) })
        : part.startsWith('$') ? tex(part.slice(1, -1))
          : part);
}

const LINK_KIND = { wiki: 'Wikipedia', paper: 'paper', video: 'video', article: 'article' };

function infoPanel(id) {
  const info = WALK_INFO[id];
  if (!info) return null;
  return el('aside', { class: 'winfo' },
    info.p.map((t) => el('p', {}, infoText(t))),
    info.links && info.links.length ? el('div', { class: 'wlinks' },
      el('h5', { text: 'read or watch' }),
      el('ul', {}, info.links.map(([kind, label, url]) => el('li', {},
        el('span', { class: 'wkind wkind-' + kind, text: LINK_KIND[kind] }),
        el('a', { href: url, target: '_blank', rel: 'noopener noreferrer', text: label }))))) : null);
}
