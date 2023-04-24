# RMT5-HYENA
RMT t5 + opimized heyena


Spent some time back engineering the described algorythm in https://arxiv.org/pdf/2304.11062.pdf to adapt it to the encoder and decoder layer of a t5, by adding positional embedding layers to the t5 decoder, it appears to be able to scale linearly and avoid the issue that Transformersxl ran into.

addiitonally got the hyene architecture from https://arxiv.org/pdf/2302.10866.pdf for modifications.
