#Example Usecase: source aggregate.sh {model_name}_{number} {patience}
#For calculating Average loss and Average acc of validation prediction
total_loss=0
total_acc=0

for k in 1 2 3 4 5 6 7 8 9 10
do
    echo "${k}-fold" | tr '\n' ': ' >> all.log
    tail -n $(($2 + 1)) $1/${k}-fold/log.log | head -n 1 >> $1/all.log
    tail -n $(($2 + 1)) $1/${k}-fold/log.log | head -n 1 | grep -o "[[:digit:]]\.[[:digit:]][[:digit:]][[:digit:]]" | tail -n 2 | tr '\n' ' ' | read loss acc
    total_loss=$(($total_loss + $loss))
    total_acc=$(($total_acc + $acc))
done
total_loss=$(($total_loss/10))
total_acc=$(($total_acc/10))
echo "total loss: ${total_loss} total_acc: ${total_acc}" >> $1/all.log
cat $1/all.log