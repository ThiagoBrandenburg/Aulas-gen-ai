# Aulas-gen-ai
Conteúdo para aulas de IA Generativa para turma de IA 2025-01 do BCC-UDESC.

# Questionários

- Aula 2: LLMs
  - https://docs.google.com/forms/d/e/1FAIpQLSfDkHmoLD_TCuoqgEHFc0xBQx7SJYNaxGHWVLyk8PWKPeiYwQ/viewform?usp=sharing&ouid=103548109541674044117

- Aula 3: Transformers
  - https://docs.google.com/forms/d/e/1FAIpQLSf7qw7mfZ-LnpMHCLpAC2mAi0SweFM2sEtxprYHDoJH1sJIHA/viewform?usp=sharing&ouid=103548109541674044117
  - Código Mini-GPT com Keras: https://keras.io/examples/generative/text_generation_with_miniature_gpt/
  - Código do DeepSeek-V3: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

- Aula 4: Fine-tuning
  - https://docs.google.com/forms/d/e/1FAIpQLSe5pgkVCqeLqz7Is8iaw3K9SZrxRzItpPMcIQpWAWCri2tUMQ/viewform?usp=sharing&ouid=103548109541674044117
  - Código em Collab: https://colab.research.google.com/drive/1sN1zHKNJa0R5pAV4H-QbvO6xIfee5VoO?usp=sharing

- Aula 5: Modelos de Imagem:
  - https://docs.google.com/forms/d/e/1FAIpQLSddamzw16lc_qyMYk229AlhI408Zf0rPFAvgIERhCXjPqVPpw/viewform?usp=sharing&ouid=103548109541674044117
  - Vídeo Computerphile: https://youtu.be/1CIpzeNxIhU

# Referências:


## Livros:
- David Foster. **Generative Deep Learning**, 2ª edição.
  - Repositório do livro: https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition
- Sebastian Raschka. **Build a Large Language Model (From Scratch)**. Simon and Schuster, 2024.
  - Repositório do livro: https://github.com/rasbt/LLMs-from-scratch


## Aula 1: Introdução
- Fundational Models:
  - Artigo do BRAINS sobre modelos fundacionais: https://brains.dev/2023/foundation-models-modelos-que-revolucionaram-a-ia/
  - Artigo do Scribble sobre modelos fundacionais: https://www.scribbledata.io/blog/foundation-models-101-a-step-by-step-guide-for-beginners/
  - Artigo da NVIDIA sobre Deep Learning: https://developer.nvidia.com/deep-learning

- Tokenização:
  - Tokenizador do ChatGPT: https://platform.openai.com/tokenizer
  - Artigo sobre a tokenização de imagens do TikTok: https://arxiv.org/abs/2406.07550
  - Artigo do Gen AI Guidebook sobre tokenização de imagens: https://ravinkumar.com/GenAiGuidebook/image/image_tokenization.html

- Embedding:
  - Artigo do BRAINS sobre tokenização e embedding:
  - Parte 1: https://brains.dev/2024/token-e-embedding-conceitos-da-ia-e-llms/
  - Parte 2: https://brains.dev/2024/embeddings-medidas-de-distancia-e-similaridade/
  - Artigo do Huggingface sobre Embeddings: https://huggingface.co/spaces/hesamation/primer-llm-embedding?section=bert_(bidirectional_encoder_representations_from_transformers)

- Modelos Fundacionais:
  - Relatório Anual da Stanford University com dados estatísticos sobre GenAis: https://hai.stanford.edu/ai-index/2025-ai-index-report
 
## Aula 2: Modelos de Geração de texto.
  - Fonte do diagrama de linha temporal de NLP: https://blog.dataiku.com/nlp-metamorphosis
  - Artigo sobre Fine-tuning: https://arxiv.org/abs/1909.08593
  - Vídeo Maximally Bad Output: https://youtu.be/qV_rOlHjvvs


## Aula 3: Transformers.
  - Embeddings de tokens:
    - Um artigo sobre interpretabilidade do espaço latente em modelos de imagem: https://arxiv.org/abs/2303.11073

  - Embeddings de posição:
    - Artigo do Hugging Face sobre Positional embedding: https://huggingface.co/blog/designing-positional-encoding
    - Artigo do Machine Learning Mastery: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/

  - Transformer:
    - Arquitetura Transformer (Attention is all you need, 2017): https://arxiv.org/abs/1706.03762
    - Arquitetura Decoder-Only (GPT, 2018): https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
    - Playlist do 3Blue1Brown sobre ANNs: https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=FFqOytYW0VzStQCz
      - As aulas 5, 6 e 7 falam sobre LLMs, são extremamente explicativas.
    - Representação visual de uma LLM: https://bbycroft.net/llm
    
  - Mixture of Experts (MoE):
    - Vídeo curto da IBM: https://youtu.be/sYDlVVyJYn4
    - Artigo sobre MoE em Transformers(Moe): https://arxiv.org/pdf/2006.16668
    - WebArticle DataCamp MoE: https://www.datacamp.com/blog/mixture-of-experts-moe
    - Artigo Deepseek-R1: https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf
    - Artigo Deepseek-V3: https://arxiv.org/pdf/2412.19437

## Aula 4: Modelos Fundacionais e Fine-tuning
  - Comparação de LLMs: https://llm-stats.com
  - Prompting: https://hub.asimov.academy/tutorial/zero-one-e-few-shot-prompts-entendendo-os-conceitos-basicos/
  - Fine tuning:
    - Fonte Diagrama fine-tuning: https://python.plainenglish.io/fine-tune-llms-a-comprehensive-guide-between-full-partial-fine-tuning-an-end-to-end-python-3fa7223f5519
    - Fine tuning your first LLM: https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face
  - Quantization:
    - https://huggingface.co/docs/optimum/concept_guides/quantization
  - LoRA:
    - Artigo original: https://arxiv.org/abs/2106.09685
  - Modelos:
    - Microsoft Phi-3 mini 4k Instruct: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

## Aula 5: Modelos de Imagem.
  - Image-to-Image:
    - Pix2Pix: https://huggingface.co/tasks/image-to-text
    - Pix2Pix Github: https://github.com/phillipi/pix2pix?tab=readme-ov-file
    - Exemplo GAN Pix2Pix: https://www.researchgate.net/publication/332932603_Transferring_Multiscale_Map_Styles_Using_Generative_Adversarial_Networks
    - Artigo Original GAN Pix2Pix: https://arxiv.org/pdf/1611.07004
    - Código geração de imagens com pix2pix: https://github.com/acucenarodrigues1998/TutorialPyBR2023-GenAINotebooks/blob/main/Tutorial_PyBR_02_Gera%C3%A7%C3%A3o_de_imagens_com_Pix2Pix.ipynb
  - Image-to-Text:
    - https://huggingface.co/tasks/image-to-text
    - https://huggingface.co/docs/transformers/model_doc/blip
  - Text-to-Image:
    - Artigo Original CLIP: https://arxiv.org/pdf/2204.06125
    - Video Computerphile sobre CLIP: https://youtu.be/KcSXcpluDe4
    - Artigo OpenAI Blip: https://openai.com/index/clip/
    - Artigo Stable Diffusion: https://arxiv.org/abs/2112.10752
  -Códigos:
  - Repositório com exemplos em português: https://github.com/acucenarodrigues1998/TutorialPyBR2023-GenAINotebooks/tree/main
  - Link para o dataset: https://huggingface.co/datasets/ashis-palai/sprites_image_dataset/blob/main/sprites_1788_16x16.npy