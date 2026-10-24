"""One ID/point-order definition shared by datasets and model adapters."""

CLASS_NAMES_EN = ['sentry', 'hero', 'engineer', 'infantry_3', 'infantry_4',
                  'infantry_5', 'outpost', 'base_bottom', 'base_top']
CLASS_NAMES_ZH = ['哨兵', '英雄', '工程', '3号步兵', '4号步兵', '5号步兵',
                  '前哨站', '基地（底部）', '基地（顶部）']
COLOR_NAMES_EN = ['blue', 'red', 'white', 'purple']
COLOR_NAMES_ZH = ['蓝色', '红色', '白色', '紫色']
POINT_NAMES = ('LT', 'LB', 'RB', 'RT')
FLIP_IDX = [3, 2, 1, 0]
OUTPUT_CHANNELS = 21
