Labels Mapping
=====

We use the 20 most frequent emojis in English. To simplify the task (and avoid unicode decoding problems) we map the emojis to numbers from 0 to 19. The mapping can be found in the txt file "us_emoji2label_mapping.txt", where each line is in the format:

label number [0:19] \<TAB\> emoji unicode \<TAB\> emoji CLDR short name

You can also check this page, were we link the images of the emojis to visualize them (some OS do not support emojis):

* [English mapping](https://fvancesco.github.io/tmp/labels_us.html)
