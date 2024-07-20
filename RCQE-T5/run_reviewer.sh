nohup python code_reviewer.py --max_source_length 200 \
  --max_target_length 200 \
  --train_batch_size 6 \
  --learning_rate 3e-4 \
  --gradient_accumulation_steps 3 \
  --seed 2233 \
  --do_train --do_test > train_reviewer.log 2>&1 &