%%writefile gpu_usage.sh
#! /bin/bash
#comment: run for 10 seconds, change it as per your use
end=$((SECONDS+10))

while [ true ]; do
# while [ $SECONDS -lt $end ]; do
    nvidia-smi --format=csv --query-gpu=power.draw,utilization.gpu,memory.used,memory.free,fan.speed,temperature.gpu -f gpu.csv
    #comment: or use below command and comment above using #
    #nvidia-smi dmon -i 0 -s mu -d 1 -o TD >> gpu.log
done