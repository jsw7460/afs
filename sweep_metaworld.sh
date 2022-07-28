# Run over single task
python3 mt_sopt_train.py \
  -m "domain_name_list=\
  [drawer-close-v2],\
  [reach-v2],\
  [window-close-v2],\
  [window-open-v2],\
  [button-press-topdown-v2],\
  [door-open-v2],\
  [drawer-open-v2],\
  [pick-place-v2],\
  [peg-insert-side-v2],\
  [push-v2]" \
  n_rl_steps=500_000
