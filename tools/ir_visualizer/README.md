# ir_visualizer
This repository includes IR visualizer which is immigrated from NPUCompiler project.

## Prerequisites
1. Python >= 3.5

2. Flatbuffers v1.11.0

3. graphviz
```
$ apt -y install graphviz
$ pip install graphviz
```
4. numpy
```
$ pip install numpy
```


## Usage
Run ir_visualizer.sh which is located in NNCompiler/scripts/tools/.

```bash
$ ./ir_visualizer.sh

<Usage> ir_visualizer.sh <command> <argument>

        <command>           <argument>

        --ir(-i)            input ir file name

```

## Visualizer Configuration

See visualizer.py --help for detailed usage examples

### General Values

|     Key     |                   Values                    | Default | desc                                                             |
|:-----------:|:-------------------------------------------:|:-------:|:-----------------------------------------------------------------|
|   engine    |        "dot", "circo", "neato", ...         |  "dot"  | *(DO NOT MODIFY)* Graph rendering engine.                        |
|   format    |                "svg", "png"                 |  "svg"  | Visualizer output file format                                    |
|  rank_dir   |                 "LR", "TB"                  |  "LR"   | Graph Direction Left-Right / Top-Bottom(not supported yet)       |
| hide_blobs  |                   Boolean                   |  false  | If set true, All blobs(kernel, bias, edge) will not be rendered. |
| hide_instr  |                   Boolean                   |  false  | If set true, All instructions will not be rendered.              |
|    node     |        {[Node Values](#Node-Values)}        |    -    | See [Node Values](#Node-Values) Below                            |
|    edge     |        {[Edge Values](#Edge-Values)}        |    -    | See [Edge Values](#Edge-Values) Below                            |
| kernel_blob | {[Kernel Blob Values](#Kernel-Blob-Values)} |    -    | See [Kernel Blob Values](#Kernel-Blob-Values) Below              |
|  bias_blob  |   {[Bias Blob Values](#Bias-Blob-Values)}   |    -    | See [Bias Blob Values](#Bias-Blob-Values) Below                  |
|  edge_blob  |   {[Edge Blob Values](#Edge-Blob-Values)}   |    -    | See [Edge Blob Values](#Edge-Blob-Values) Below                  |

### Node Values

|    Key     |                                Values                                 | Default | desc                   |
|:----------:|:---------------------------------------------------------------------:|:-------:|:-----------------------|
| font_size  |                                  Num                                  |   10    | Set font size of Edge  |
| font_color | [Color Names](https://graphviz.gitlab.io/_pages/doc/info/colors.html) | "black" | Set font color of Edge |

### Edge Values

|    Key     |                                Values                                 | Default | desc                   |
|:----------:|:---------------------------------------------------------------------:|:-------:|:-----------------------|
| font_size  |                                  Num                                  |   10    | Set font size of Edge  |
| font_color | [Color Names](https://graphviz.gitlab.io/_pages/doc/info/colors.html) | "black" | Set font color of Edge |
|   color    | [Color Names](https://graphviz.gitlab.io/_pages/doc/info/colors.html) | "white" | Set fill color of Edge |

### Kernel Blob Values

|    Key     |                                Values                                 |  Default  | desc                          |
|:----------:|:---------------------------------------------------------------------:|:---------:|:------------------------------|
| font_size  |                                  Num                                  |     9     | Set font size of Kernel Blob  |
| font_color | [Color Names](https://graphviz.gitlab.io/_pages/doc/info/colors.html) |  "white"  | Set font color of Kernel Blob |
|   color    | [Color Names](https://graphviz.gitlab.io/_pages/doc/info/colors.html) | "maroon4" | Set fill color of Kernel Blob |

### Bias Blob Values

|    Key     |                                Values                                 |   Default    | desc                        |
|:----------:|:---------------------------------------------------------------------:|:------------:|:----------------------------|
| font_size  |                                  Num                                  |      9       | Set font size of Bias Blob  |
| font_color | [Color Names](https://graphviz.gitlab.io/_pages/doc/info/colors.html) |   "white"    | Set font color of Bias Blob |
|   color    | [Color Names](https://graphviz.gitlab.io/_pages/doc/info/colors.html) | "royalblue4" | Set fill color of Bias Blob |

### Edge Blob Values

|    Key     |                                Values                                 | Default | desc                                             |
|:----------:|:---------------------------------------------------------------------:|:-------:|:-------------------------------------------------|
| font_size  |                                  Num                                  |    9    | Edge Blob's font size|Set font size of Edge Blob |
| font_color | [Color Names](https://graphviz.gitlab.io/_pages/doc/info/colors.html) | "white" | Set font color of Edge Blob                      |
|   color    | [Color Names](https://graphviz.gitlab.io/_pages/doc/info/colors.html) | "black" | Set fill color of Edge Blob                      |
