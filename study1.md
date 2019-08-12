Ablations Study done with patch_size=32, glimpse_scale=2 for pretrained pytorch models with vanilla RNN
|Model        |Num Patches|Accuracy|           Confusion Matrix              |
|-------------|----------:|-------:|-----------------------------------------|
|Resnet18     |          1|        |                                         |
|Resnet18     |          2|        |                                         |
|Resnet34     |          1|        |                                         |
|Resnet34     |          2|        |                                         |
|Resnet50     |          1|        |                                         |
|Resnet50     |          2|        |                                         |
|Resnet101    |          1|        |                                         |
|Resnet101    |          2|        |                                         |
|Resnet152    |          1|        |                                         |
|Resnet152    |          2|        |                                         |
|Densenet121  |          1|   90.33|[[ 82  25   0][  2  77   0][  0   2 112]]|
|Densenet121  |          2|   93.67|[[105   2   0][  6  71   2][  3   6 105]]|
|Densenet161  |          1|   91.33|[[ 90  17   0][  3  75   1][  2   3 109]]|
|Densenet161  |          2|   89.33|[[ 98   9   0][ 15  59   5][  3   0 111]]|
|Densenet169  |          1|   91.00|[[ 95  12   0][  9  66   4][  2   0 112]]|
|Densenet169  |          2|   92.33|[[ 96   8   3][  6  68   5][  0   1 113]]|
|Densenet201  |          1|   92.00|[[ 86  21   0][  1  77   1][  1   0 113]]|
|Densenet201  |          2|   93.33|[[ 94   9   4][  2  72   5][  0   0 114]]|
