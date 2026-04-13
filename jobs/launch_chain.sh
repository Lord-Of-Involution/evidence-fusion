#!/bin/bash
# 自动连环提交脚本：榨干超算碎片时间的“铁炮连击”

echo ">>> 开始装填弹药：准备发射 4 发 1 小时连环 GNN 任务..."

# 1. 确保在正确的目录下
cd /home/x-jdong8/src/evidence-fusion/jobs

# 2. 发射第一发（无依赖，先锋队）
JOB1=$(sbatch --parsable 07_train_gnn_solo.sh)
echo "🎯 第一发已升空 (Job ID: $JOB1) - 随时准备抢占空闲节点"

# 3. 发射第二发（等待第一发倒下后无缝接力）
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 07_train_gnn_solo.sh)
echo "🎯 第二发已装填 (Job ID: $JOB2) - 锁定目标 $JOB1"

# 4. 发射第三发
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 07_train_gnn_solo.sh)
echo "🎯 第三发已装填 (Job ID: $JOB3) - 锁定目标 $JOB2"

# 5. 发射第四发
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 07_train_gnn_solo.sh)
echo "🎯 第四发已装填 (Job ID: $JOB4) - 锁定目标 $JOB3"

echo ">>> 阵列部署完毕！请主君立刻合上电脑看剧！"