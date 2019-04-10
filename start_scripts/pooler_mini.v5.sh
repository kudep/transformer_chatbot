start_cmd='bash start_scripts/mini_experiments.v5.sh'
export CUDA_VISIBLE_DEVICES=1
{
$start_cmd MINI_V5_BT_L3_TW1_SEG1_V40_ShedLR0_SpecT1_CS_SpecTReinit1_EB_manSL128
}&
export CUDA_VISIBLE_DEVICES=2
{
$start_cmd MINI_V5_BT_L3_TW1_SEG0_V40_ShedLR0_SpecT1_CS_SpecTReinit1_EB_manSL128
}&
export CUDA_VISIBLE_DEVICES=3
{
$start_cmd MINI_V5_BT_L3_TW1_SEG1_V40_ShedLR0_SpecT1_S_SpecTReinit1_EB_manSL128
}&
export CUDA_VISIBLE_DEVICES=4
{
$start_cmd MINI_V5_BT_L3_TW1_SEG0_V40_ShedLR0_SpecT1_S_SpecTReinit1_EB_manSL128
}&
