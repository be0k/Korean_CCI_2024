# 리스트를 반복하여 사용
for i in {0..10}
do
    if [ $i -eq 10 ]; then
        param=1890
    else
        param=2040
    fi
   python -m run.test \
   --output output/fold$i.json \
    --model_id kihoonlee/STOCK_SOLAR-10.7B \
    --tokenizer kihoonlee/STOCK_SOLAR-10.7B \
    --device cuda \
    --peft_model_dir ./test_git/checkpoint-$param \
    --test_dir data/test.json
done


python -m run.ensemble \
--dir output \
--d0 fold0.json \
--d1 fold1.json \
--d2 fold2.json \
--d3 fold3.json \
--d4 fold4.json \
--d5 fold5.json \
--d6 fold6.json \
--d7 fold7.json \
--d8 fold8.json \
--d9 fold9.json \
--d10 fold10.json \
--save_dir ensemble.json
