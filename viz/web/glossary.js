'use strict';
// The glossary shown in "About the model". Each entry: term, other names it
// goes by (for search), group, an explanation in terms of this model, links
// for digging deeper, and optionally the tab of this page where you can see it.

const W = (page) => ['Wikipedia: ' + page.replace(/_/g, ' '), 'https://en.wikipedia.org/wiki/' + page];
const ARXIV = (id, title) => [title, 'https://arxiv.org/abs/' + id];

const GLOSSARY_GROUPS = [
  'Text and tokens',
  'Inside the model',
  'Attention',
  'Running the model',
  'Output and sampling',
  'Looking inside',
  'Training',
];

const GLOSSARY = [
  // ------------------------------------------------------------ text and tokens
  {
    t: 'Token', aka: 'token id', g: 'Text and tokens',
    d: 'The unit a language model reads and writes: a word, part of a word, a punctuation mark or a single byte, each with an id. SmolLM2 knows 49,152 of them. "That" is one token (id 5195), " That" with a leading space is another (1848), and "That\'s" is two: "That" + "\'s". Models never see letters, only these ids.',
    l: [['Hugging Face: tokenizers explained', 'https://huggingface.co/docs/transformers/main/en/tokenizer_summary']],
  },
  {
    t: 'Tokenizer', aka: 'tokenization encode decode', g: 'Text and tokens',
    d: 'The fixed program that turns text into token ids and back. It isn\'t part of the neural network and isn\'t learned by gradient descent; it\'s built once from a corpus before training. Here it runs in two stages: a pre-tokenizer splits text into word-like pieces (and every digit on its own), then BPE merges each piece\'s bytes into tokens.',
    l: [['Hugging Face course: tokenizers', 'https://huggingface.co/learn/llm-course/chapter6/5']],
  },
  {
    t: 'Byte-pair encoding (BPE)', aka: 'bpe merges subword', g: 'Text and tokens',
    d: 'How the vocabulary is built and applied. Start from single bytes and repeatedly glue together the most frequent adjacent pair, recording each merge. Encoding replays those merges in order: T h a t → T h at → T hat → That. SmolLM2 has 48,900 merges. Byte-level means it starts from raw UTF-8 bytes, so any text can be encoded and there is no "unknown" token.',
    l: [W('Byte-pair_encoding'), ARXIV('1508.07909', 'Sennrich et al. 2016, BPE for neural models'), ['Hugging Face course: BPE', 'https://huggingface.co/learn/llm-course/chapter6/5']],
  },
  {
    t: 'Vocabulary', aka: 'vocab', g: 'Text and tokens',
    d: 'The set of all tokens the model can read or produce: 49,152 here. The size is a trade-off: a bigger vocabulary makes text shorter in tokens but makes the embedding table and the output layer bigger. In a model this small the embedding table is already 21% of all parameters.',
    l: [['Hugging Face: tokenizer summary', 'https://huggingface.co/docs/transformers/main/en/tokenizer_summary']],
  },
  {
    t: 'Special tokens', aka: 'endoftext bos eos control tokens', g: 'Text and tokens',
    d: 'Tokens that never come from ordinary text and mark structure instead. SmolLM2 has 17 (ids 0–16), mostly for code (<file_sep>, <jupyter_code>…). The important one here is <|endoftext|> (0): in training it separates documents, so when the model produces it, it considers the text finished and generation stops.',
    l: [['Hugging Face: tokenizer summary', 'https://huggingface.co/docs/transformers/main/en/tokenizer_summary']],
  },
  {
    t: 'Base model vs chat model', aka: 'base instruct chat template chatml completion', g: 'Text and tokens',
    d: 'This page runs SmolLM2-135M, a base model: it only continues text, so the prompt is the start of a text, not a question. A chat model (SmolLM2-135M-Instruct) is the same network trained further on conversations written in a template like "<|im_start|>user … <|im_end|> <|im_start|>assistant"; chatting with it is still text completion, of that template.',
    l: [['Hugging Face: chat templates', 'https://huggingface.co/docs/transformers/main/en/chat_templating']],
  },
  {
    t: 'UTF-8', aka: 'bytes unicode', g: 'Text and tokens',
    d: 'The standard way of storing text as bytes: 1 byte for ASCII letters, 2–4 for other characters and emoji. Because tokens are byte sequences, one character can be split across tokens, and a token on its own may not be valid text. That is why a streamed reply sometimes waits a token before showing a character.',
    l: [W('UTF-8')],
  },

  // ------------------------------------------------------------ inside the model
  {
    t: 'Parameters', aka: 'weights params', g: 'Inside the model',
    d: 'The learned numbers of the network, fixed after training: 134,515,008 of them here, stored as 269 MB of bfloat16. Almost all live in weight matrices. "135M" in the name is this count. Running the model only reads them; nothing changes them during a conversation.',
    l: [W('Large_language_model'), ['3Blue1Brown: neural networks', 'https://www.3blue1brown.com/topics/neural-networks']],
  },
  {
    t: 'Transformer', aka: 'architecture', g: 'Inside the model',
    d: 'The neural network design behind nearly all modern language models (2017). A stack of identical layers, each made of an attention step (positions exchange information) and an MLP step (each position is processed on its own), wrapped in residual connections and normalization. SmolLM2 follows the Llama variant of it.',
    l: [W('Transformer_(deep_learning_architecture)'), ARXIV('1706.03762', 'Vaswani et al. 2017, Attention Is All You Need'), ['The Illustrated Transformer', 'https://jalammar.github.io/illustrated-transformer/']],
  },
  {
    t: 'Decoder-only, autoregressive', aka: 'causal language model gpt next-token prediction', g: 'Inside the model',
    d: 'A model that does one thing: given tokens so far, give a probability for every possible next token. Text is generated by picking one, appending it and asking again (autoregression). GPT, Llama and SmolLM2 are all decoder-only; the original Transformer also had an encoder for translation.',
    l: [W('Autoregressive_model'), W('Generative_pre-trained_transformer'), ARXIV('2005.14165', 'Brown et al. 2020, GPT-3')],
  },
  {
    t: 'Embedding', aka: 'embed embedding table word vector', g: 'Inside the model',
    d: 'The first step: each token id selects one row of a 49,152 × 576 table, giving a vector of 576 numbers. Training places tokens with similar uses close together: the nearest rows to "That" are " That", "that", " THAT" and "Those". The vector is the model\'s starting point, before any context has been mixed in.',
    l: [W('Word_embedding')], tab: 'walk',
  },
  {
    t: 'Hidden size (d_model)', aka: 'width model dimension 576', g: 'Inside the model',
    d: 'The length of the vector the model keeps for every position: 576 here (GPT-3 uses 12,288). Every layer reads and writes vectors of this size, so it bounds how much the model can hold about one position at a time.',
    l: [['The Illustrated Transformer', 'https://jalammar.github.io/illustrated-transformer/']],
  },
  {
    t: 'Residual stream', aka: 'residual connection skip connection hidden state', g: 'Inside the model',
    d: 'The running vector per position that flows up through the model. Layers never replace it; each one reads it and adds its output back (x = x + layer(x)). So the final vector is the embedding plus the sum of everything every layer wrote, and you can measure each layer\'s contribution.',
    l: [W('Residual_neural_network'), ['Anthropic: A Mathematical Framework for Transformer Circuits', 'https://transformer-circuits.pub/2021/framework/index.html']], tab: 'walk',
  },
  {
    t: 'Layer (block)', aka: 'transformer block decoder layer', g: 'Inside the model',
    d: 'One repeat of the basic unit: norm → attention → add, then norm → MLP → add. SmolLM2 stacks 30 of them, each with its own weights (3.54M parameters per layer). Early layers tend to handle local, surface patterns, later ones more abstract ones, but nothing forces a neat division.',
    l: [W('Transformer_(deep_learning_architecture)')], tab: 'walk',
  },
  {
    t: 'RMSNorm', aka: 'normalization layer norm layernorm pre-norm', g: 'Inside the model',
    d: 'Rescales a vector so its root-mean-square is 1, then multiplies each coordinate by a learned weight. It keeps the input of every sublayer at a predictable scale however big the residual stream grows. "Pre-norm" means it\'s applied to the input of each sublayer, not to the stream itself, which makes deep models much easier to train.',
    l: [ARXIV('1910.07467', 'Zhang & Sennrich 2019, RMSNorm'), ARXIV('2002.04745', 'Xiong et al. 2020, pre-norm vs post-norm'), W('Normalization_(machine_learning)')], tab: 'walk',
  },
  {
    t: 'MLP (feed-forward network)', aka: 'ffn feedforward mlp', g: 'Inside the model',
    d: 'The second half of each layer, applied to every position separately: project the 576-vector up to 1,536 numbers, apply a nonlinearity, project back down. It holds 59% of SmolLM2\'s parameters and is thought to store much of what the model "knows", like lookup tables from patterns to facts.',
    l: [W('Multilayer_perceptron')], tab: 'walk',
  },
  {
    t: 'SwiGLU', aka: 'glu gated linear unit gate up down', g: 'Inside the model',
    d: 'The MLP variant used here: down( silu(gate·x) ⊙ up·x ). Two separate projections up to 1,536; one goes through SiLU and multiplies the other element by element, so one half decides how much of the other passes. It needs three matrices instead of two but trains better for the same compute.',
    l: [ARXIV('2002.05202', 'Shazeer 2020, GLU Variants Improve Transformer'), ARXIV('1612.08083', 'Dauphin et al. 2016, gated linear units')], tab: 'walk',
  },
  {
    t: 'SiLU (swish)', aka: 'activation function nonlinearity relu', g: 'Inside the model',
    d: 'The nonlinearity in the MLP: silu(x) = x · sigmoid(x). Roughly x for large positive inputs and roughly 0 for large negative ones, with a smooth dip in between. Without some nonlinearity the whole network would collapse into a single matrix multiplication.',
    l: [W('Swish_function'), W('Activation_function')], tab: 'walk',
  },
  {
    t: 'Neuron', aka: 'mlp neuron unit', g: 'Inside the model',
    d: 'One of the 1,536 intermediate numbers in an MLP, and the column of weights that writes it back to the residual stream. Some neurons are tied to clear features, but most respond to mixtures. Two neurons in layer 11 are the ones that create the huge first-token activation.',
    l: [['Anthropic: A Mathematical Framework for Transformer Circuits', 'https://transformer-circuits.pub/2021/framework/index.html']], tab: 'walk',
  },
  {
    t: 'Tied embeddings', aka: 'weight tying lm head unembedding output head', g: 'Inside the model',
    d: 'The output layer reuses the embedding table: the score of a token is the dot product of the final vector with that token\'s embedding row. This saves 28.3M parameters (21% of this model) and means "predicting token X" and "reading token X" share one vector.',
    l: [ARXIV('1608.05859', 'Press & Wolf 2016, Using the Output Embedding')], tab: 'walk',
  },

  // ------------------------------------------------------------ attention
  {
    t: 'Attention', aka: 'self-attention scaled dot-product', g: 'Attention',
    d: 'The only step where positions exchange information. Each position makes a query; every earlier position offers a key and a value. Query·key scores, after softmax, become weights that add up to 1, and the output is the weighted sum of values. "Look back at whatever is relevant and copy some of it here."',
    l: [W('Attention_(machine_learning)'), ['3Blue1Brown: attention', 'https://www.3blue1brown.com/lessons/attention'], ARXIV('1706.03762', 'Vaswani et al. 2017')], tab: 'walk',
  },
  {
    t: 'Query, key, value (q, k, v)', aka: 'qkv projections', g: 'Attention',
    d: 'Three different projections of the same (normalized) residual vector. Query: what this position is looking for. Key: what this position offers, to be matched against queries. Value: what it hands over when attended to. Each is 64 numbers per head. Scores are q·k / √64, so the scale doesn\'t grow with the size.',
    l: [W('Attention_(machine_learning)'), ['The Illustrated Transformer', 'https://jalammar.github.io/illustrated-transformer/']], tab: 'walk',
  },
  {
    t: 'Attention head', aka: 'multi-head attention heads', g: 'Attention',
    d: 'Attention is run several times in parallel with separate weights; each copy is a head and can learn its own pattern (previous token, same word earlier, start of sentence…). SmolLM2 has 9 heads of 64 dimensions per layer, 270 in total. Their outputs are concatenated and projected back to 576.',
    l: [W('Transformer_(deep_learning_architecture)')], tab: 'walk',
  },
  {
    t: 'Grouped-query attention (GQA)', aka: 'gqa multi-query mqa kv heads', g: 'Attention',
    d: 'Heads share keys and values: SmolLM2\'s 9 query heads are split into 3 groups, each using one key/value head. The KV cache is 3× smaller and decoding is faster, at little cost in quality. The extreme, one shared key/value head, is multi-query attention.',
    l: [ARXIV('2305.13245', 'Ainslie et al. 2023, GQA'), ARXIV('1911.02150', 'Shazeer 2019, multi-query attention')], tab: 'walk',
  },
  {
    t: 'Causal mask', aka: 'masking autoregressive mask', g: 'Attention',
    d: 'Position i may only attend to positions 0…i, never to later ones: future scores are set to −∞ before the softmax. That is what lets the model be trained on every position of a text at once, and why the attention heatmaps are triangles.',
    l: [W('Transformer_(deep_learning_architecture)')], tab: 'walk',
  },
  {
    t: 'RoPE (rotary position embedding)', aka: 'rotary positional encoding position', g: 'Attention',
    d: 'Attention by itself doesn\'t know word order. RoPE encodes it by rotating pairs of numbers in q and k by an angle proportional to the position, at 32 different speeds (base θ = 100,000). The rotation cancels out in q·k except for the difference in positions, so heads see relative distance. At position 0 the angle is 0: nothing happens.',
    l: [ARXIV('2104.09864', 'Su et al. 2021, RoFormer'), ['EleutherAI: rotary embeddings explained', 'https://blog.eleuther.ai/rotary-embeddings/']], tab: 'walk',
  },
  {
    t: 'Attention sink', aka: 'sink first token bos', g: 'Attention',
    d: 'Softmax weights must add up to 1, so a head with nothing useful to read still has to put its attention somewhere. Models learn to dump it on the first token, which then works as a no-op. That is why so many heatmaps have a bright first column, and why a lone first token makes poor predictions: its vector is busy being the sink.',
    l: [ARXIV('2309.17453', 'Xiao et al. 2023, attention sinks')], tab: 'walk',
  },
  {
    t: 'Massive activations', aka: 'outlier dimensions spike', g: 'Attention',
    d: 'A handful of residual-stream coordinates that become huge (thousands of times the rest) at the first token. In SmolLM2 two neurons in layer 11 write about 22,500 into dimension 507, and layers 28–29 remove it again. It works like a fixed bias that makes the attention sink possible.',
    l: [ARXIV('2402.17762', 'Sun et al. 2024, Massive Activations in LLMs')], tab: 'walk',
  },

  // ------------------------------------------------------------ running the model
  {
    t: 'Forward pass', aka: 'inference', g: 'Running the model',
    d: 'One run of the network from token ids to next-token scores: embed, 30 layers, final norm, output head. It is pure arithmetic with fixed weights, deterministic, and the same in this browser as on a GPU. Generating text is a loop of forward passes plus a sampling step.',
    l: [['Hugging Face: generation with LLMs', 'https://huggingface.co/docs/transformers/main/en/llm_tutorial']], tab: 'walk',
  },
  {
    t: 'Prefill', aka: 'prompt processing', g: 'Running the model',
    d: 'The first forward pass, over the whole prompt at once. Every position goes through the layers in parallel as big matrix multiplications, so it\'s compute-bound and fast per token. It fills the KV cache and gives the prediction for the first new token.',
    l: [['Hugging Face: KV cache', 'https://huggingface.co/docs/transformers/main/en/kv_cache']],
  },
  {
    t: 'Decode', aka: 'decoding generation step', g: 'Running the model',
    d: 'Each later step feeds exactly one new token and reads everything else from the KV cache. Every step has to stream all 135M weights from memory just to process one vector, so it is memory-bound: you get tens of tokens per second, while prefill handles hundreds at once.',
    l: [['Hugging Face: KV cache', 'https://huggingface.co/docs/transformers/main/en/kv_cache'], W('Roofline_model')],
  },
  {
    t: 'KV cache', aka: 'key value cache', g: 'Running the model',
    d: 'Keys and values of past positions never change, so they are stored instead of recomputed. Each new token then costs one position\'s work, not the whole prefix again. Here: 30 layers × (k + v) × 3 heads × 64 numbers × 4 bytes = 46 KB per token. Rewinding generation just truncates this cache.',
    l: [['Hugging Face: KV cache', 'https://huggingface.co/docs/transformers/main/en/kv_cache']],
  },
  {
    t: 'Context window', aka: 'context length max positions', g: 'Running the model',
    d: 'How many tokens the model can take into account at once. SmolLM2 was trained on 8,192; this page uses 512, because it keeps every activation of every position. Anything beyond the window simply doesn\'t exist for the model.',
    l: [W('Large_language_model')],
  },
  {
    t: 'Matrix multiplication', aka: 'matmul linear layer projection', g: 'Running the model',
    d: 'Almost all of the work. Every projection (q, k, v, o, gate, up, down, output head) multiplies a vector by a weight matrix: about 2 floating-point operations per parameter per token, so ~270 million per token here. This engine has hand-written SIMD kernels for it.',
    l: [W('Matrix_multiplication')],
  },
  {
    t: 'Memory-bound vs compute-bound', aka: 'roofline bandwidth', g: 'Running the model',
    d: 'A step is memory-bound when moving data takes longer than the arithmetic, compute-bound when the reverse is true. Decoding one token uses each weight once after loading it, so memory bandwidth limits speed. Prefill reuses each loaded weight for many positions, so arithmetic does.',
    l: [W('Roofline_model')],
  },
  {
    t: 'bfloat16 and float32', aka: 'bf16 fp32 f32 precision dtype', g: 'Running the model',
    d: 'Number formats. bfloat16 uses 2 bytes: the range of a 32-bit float with only about 3 significant digits, good enough for weights. SmolLM2 is published in bf16 (269 MB); this engine converts it to 4-byte float32 (538 MB) and does all its arithmetic in f32, which is why it matches Hugging Face to within 1e-4.',
    l: [W('Bfloat16_floating-point_format'), W('Single-precision_floating-point_format')],
  },
  {
    t: 'Safetensors', aka: 'model file weights format', g: 'Running the model',
    d: 'The file format of the weights: an 8-byte length, a JSON header naming every tensor with its shape and byte range, then the raw numbers. Unlike Python pickles it can\'t run code when loaded, and it can be read without parsing everything.',
    l: [['Hugging Face: safetensors', 'https://huggingface.co/docs/safetensors/index']],
  },
  {
    t: 'WebAssembly and SIMD', aka: 'wasm simd128 neon', g: 'Running the model',
    d: 'WebAssembly is a compact binary format browsers run at near-native speed; the Rust engine is compiled to it. SIMD (single instruction, multiple data) processes several numbers per instruction, 4 floats at a time with wasm-simd128 or ARM NEON, which is where most of the speed comes from.',
    l: [W('WebAssembly'), W('Single_instruction,_multiple_data')],
  },

  // ------------------------------------------------------------ output and sampling
  {
    t: 'Logits', aka: 'scores unnormalized', g: 'Output and sampling',
    d: 'The raw output: one score per vocabulary token, 49,152 numbers, from a dot product of the final vector with each token\'s embedding. Only differences matter: adding the same amount to all of them changes nothing. A difference of 1 means a factor of e ≈ 2.7 in probability.',
    l: [W('Logit')], tab: 'next',
  },
  {
    t: 'Softmax', aka: 'probabilities normalize', g: 'Output and sampling',
    d: 'Turns scores into probabilities: exponentiate each, divide by the sum. Bigger scores get exponentially more weight, and the results are positive and add up to 1. Used twice in this model: over the vocabulary at the end, and over positions inside every attention head.',
    l: [W('Softmax_function')], tab: 'next',
  },
  {
    t: 'Log-sum-exp', aka: 'lse partition function normalizer', g: 'Output and sampling',
    d: 'log Σ exp(logits): the number subtracted from every logit to turn it into a log-probability. A token\'s probability is exp(logit − lse). The Next token view shows it.',
    l: [W('LogSumExp')], tab: 'next',
  },
  {
    t: 'Sampling', aka: 'decoding strategy generation', g: 'Output and sampling',
    d: 'Choosing the actual next token from the probabilities. The model only provides the distribution; the sampler decides, after reshaping it with temperature, top-k, top-p and penalties. Different settings turn the same model into a careful or a creative writer.',
    l: [['Hugging Face: how to generate text', 'https://huggingface.co/blog/how-to-generate'], ['Hugging Face: generation strategies', 'https://huggingface.co/docs/transformers/main/en/generation_strategies']], tab: 'next',
  },
  {
    t: 'Greedy decoding', aka: 'argmax temperature 0', g: 'Output and sampling',
    d: 'Always take the single most likely token (temperature 0 here). Deterministic, and good for short factual answers, but long texts tend to get stuck in loops, especially with small models.',
    l: [['Hugging Face: how to generate text', 'https://huggingface.co/blog/how-to-generate']],
  },
  {
    t: 'Temperature', aka: 'temp', g: 'Output and sampling',
    d: 'Divide the logits by T before softmax. T < 1 sharpens the distribution (more predictable), T > 1 flattens it (more surprising), T → 0 is greedy. The default preset here is greedy (T = 0).',
    l: [['Hugging Face: how to generate text', 'https://huggingface.co/blog/how-to-generate'], W('Softmax_function')], tab: 'next',
  },
  {
    t: 'Top-k', aka: 'top k', g: 'Output and sampling',
    d: 'Keep only the k most likely tokens and renormalize. A blunt cutoff of the long tail of unlikely tokens; off by default here.',
    l: [['Hugging Face: how to generate text', 'https://huggingface.co/blog/how-to-generate']], tab: 'next',
  },
  {
    t: 'Top-p (nucleus sampling)', aka: 'nucleus top p', g: 'Output and sampling',
    d: 'Keep the smallest set of most likely tokens whose probabilities add up to p (0.9 here), drop the rest and renormalize. Unlike top-k it adapts: one token when the model is sure, hundreds when it isn\'t.',
    l: [ARXIV('1904.09751', 'Holtzman et al. 2019, nucleus sampling'), ['Hugging Face: how to generate text', 'https://huggingface.co/blog/how-to-generate']], tab: 'next',
  },
  {
    t: 'Repetition penalty', aka: 'repeat penalty frequency presence', g: 'Output and sampling',
    d: 'Lower the logit of every token that already appeared recently (divide a positive logit by the penalty, multiply a negative one). It fights loops, but bluntly: real text legitimately repeats words like "the" and "is", and the penalty pushes them down too. Off by default here, in favour of DRY.',
    l: [ARXIV('1909.05858', 'Keskar et al. 2019, CTRL (introduced it)')], tab: 'next',
  },
  {
    t: 'DRY ("don\'t repeat yourself")', aka: 'dry anti-loop sampler', g: 'Output and sampling',
    d: 'Penalizes only a token that would continue a sequence that already appeared earlier, and harder the longer that repeat is: multiplier × base^(length − 2). Single repeated words are free; "I went to the office. I went to the" is stopped. It is on by default here.',
    l: [['p-e-w: the original DRY proposal', 'https://github.com/oobabooga/text-generation-webui/pull/5677']], tab: 'next',
  },

  // ------------------------------------------------------------ looking inside
  {
    t: 'Activation', aka: 'hidden state intermediate tensor', g: 'Looking inside',
    d: 'Any intermediate number the model computes while running, as opposed to the fixed weights. Attention patterns, MLP neuron values and the residual stream are all activations; they depend on the input. Interpretability is mostly about reading them.',
    l: [['Anthropic: A Mathematical Framework for Transformer Circuits', 'https://transformer-circuits.pub/2021/framework/index.html']],
  },
  {
    t: 'Hook', aka: 'hooks intervention', g: 'Looking inside',
    d: 'A callback the engine calls with each activation as it is computed, which can record it or change it before the rest of the model sees it. Every view on this page is built from hooks. TransformerLens offers the same idea for PyTorch models.',
    l: [['TransformerLens', 'https://github.com/TransformerLensOrg/TransformerLens']],
  },
  {
    t: 'Logit lens', aka: 'lens early decoding', g: 'Looking inside',
    d: 'Apply the final norm and output head to the residual stream after an intermediate layer, as if the model stopped there. You see what it "would predict" at each depth, and where the answer appears. It works because every layer writes into the same stream the output head reads.',
    l: [['nostalgebraist: interpreting GPT, the logit lens', 'https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens']],
  },
  {
    t: 'Direct logit attribution', aka: 'dla logit attribution contribution write', g: 'Looking inside',
    d: 'The final logits are the unembedding applied to the normalized residual stream, and the stream is a sum of writes. Freeze the final RMSNorm\'s divisor at its real value and each write\'s effect on every logit becomes exact and additive: you can say "layer 23\'s attention added +5.4 to the logit of \' Berlin\'".',
    l: [['Anthropic: A Mathematical Framework for Transformer Circuits', 'https://transformer-circuits.pub/2021/framework/index.html'], ['nostalgebraist: interpreting GPT, the logit lens', 'https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens']], tab: 'walk',
  },
  {
    t: 'Ablation', aka: 'ablate knock out', g: 'Looking inside',
    d: 'Switch a part of the model off (here: zero an attention head\'s output) and see what changes. If the prediction collapses, that part mattered for this input.',
    l: [W('Ablation_(artificial_intelligence)')], tab: 'walk',
  },
  {
    t: 'Mechanistic interpretability', aka: 'circuits interpretability', g: 'Looking inside',
    d: 'The research field that tries to reverse-engineer what algorithms trained networks implement, from weights and activations, instead of treating them as black boxes. Hooks, the logit lens and ablation are its basic instruments.',
    l: [W('Mechanistic_interpretability'), ['Anthropic: A Mathematical Framework for Transformer Circuits', 'https://transformer-circuits.pub/2021/framework/index.html']],
  },
  {
    t: 'Dot product and cosine similarity', aka: 'similarity vector', g: 'Looking inside',
    d: 'The dot product multiplies two vectors coordinate by coordinate and adds it all up: large when they point the same way. Cosine similarity is the dot product of the vectors scaled to length 1, between −1 and 1. Attention scores, logits and "nearest embeddings" are all dot products.',
    l: [W('Dot_product'), W('Cosine_similarity')],
  },

  // ------------------------------------------------------------ training
  {
    t: 'Pretraining', aka: 'next-token prediction base model', g: 'Training',
    d: 'The main training phase: show the model huge amounts of text (2 trillion tokens for SmolLM2) and adjust all weights so that every position predicts the following token a bit better. Everything the model knows comes from this. The result is a base model that continues text but doesn\'t follow instructions.',
    l: [W('Large_language_model'), ARXIV('2502.02737', 'Allal et al. 2025, the SmolLM2 paper')],
  },
  {
    t: 'Loss (cross-entropy)', aka: 'loss function perplexity', g: 'Training',
    d: 'The number training minimizes: −log of the probability the model gave to the token that actually came next, averaged over all positions. 0 would be perfect certainty; a uniform guess over 49,152 tokens would score log 49,152 ≈ 10.8.',
    l: [W('Cross-entropy')],
  },
  {
    t: 'Gradient descent and backpropagation', aka: 'backprop training optimizer', g: 'Training',
    d: 'How weights are adjusted. Backpropagation computes, for every one of the 135M parameters, how the loss would change if it moved a little; gradient descent then moves each parameter a small step in the direction that lowers the loss. Repeat billions of times.',
    l: [W('Gradient_descent'), W('Backpropagation'), ['Karpathy: Neural Networks: Zero to Hero', 'https://karpathy.ai/zero-to-hero.html']],
  },
  {
    t: 'Supervised fine-tuning (SFT)', aka: 'instruction tuning fine-tuning instruct', g: 'Training',
    d: 'Continue training the base model on example conversations written in the chat template, so it learns to answer as an assistant and to end its turn. SmolLM2-135M-Instruct got this (on the smol-smoltalk dataset); the base model on this page did not.',
    l: [W('Fine-tuning_(deep_learning)')],
  },
  {
    t: 'DPO (direct preference optimization)', aka: 'dpo rlhf preference alignment', g: 'Training',
    d: 'A final training step on pairs of answers to the same prompt, one preferred over the other: shift probability towards the preferred kind of answer. A simpler alternative to RLHF that needs no separate reward model. SmolLM2-Instruct used the UltraFeedback dataset.',
    l: [ARXIV('2305.18290', 'Rafailov et al. 2023, DPO'), W('Reinforcement_learning_from_human_feedback')],
  },
];
