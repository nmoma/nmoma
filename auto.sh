#!/bin/bash

set -e

# ---- Check argument ----
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

EXP_NAME=$1

# ---- Paths ----
YAML_FILE="src/planner/params/agent.yaml"
OUTPUT_DIR="./src/planner/logs"
BASE_SAVE_DIR="./logs/$EXP_NAME"

mkdir -p "$BASE_SAVE_DIR"

# ---- ROS environment ----
source ./devel/setup.bash

# ---- Functions ----
rb() {
    roslaunch planner run_all.launch rviz:=false
}

d() {
    python ./diffusion_node.py
}

replica() {
    python ./task_node.py
}

rp() {
    rostopic pub -r 1 /move_base_simple/goal geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:
    x: 0.0
    y: 0.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0"
}

# ---- Main loop ----
for i in {2..6}; do
    echo "===== Running $EXP_NAME with ddim_path_num=$i ====="

    RUN_DIR="$BASE_SAVE_DIR/run_$i"
    mkdir -p "$RUN_DIR"

    # 1. Modify YAML
    sed -i "s/ddim_path_num: .*/ddim_path_num: $i/" "$YAML_FILE"

    # 2. Snapshot files
    BEFORE_FILES=$(ls "$OUTPUT_DIR"/benchmark_*.txt 2>/dev/null || true)

    # 3. Start processes (NO OUTPUT AT ALL)
    roslaunch planner run_all.launch rviz:=false > /dev/null 2>&1 &
    # roslaunch planner run_all.launch rviz:=false 2>/dev/null  &
    RB_PID=$!

    python ./diffusion_node.py > /dev/null 2>&1 &
    # python ./diffusion_node.py &
    D_PID=$!

    python ./task_node.py > /dev/null 2>&1 &
    REPLICA_PID=$!

    sleep 10

    rp > /dev/null 2>&1 &
    RP_PID=$!

    # 4. Wait for rb
    wait $RB_PID || true

    # 5. Kill others
    kill $RP_PID 2>/dev/null || true
    kill $D_PID 2>/dev/null || true
    kill $REPLICA_PID 2>/dev/null || true

    sleep 5

    # 6. Detect new file
    AFTER_FILES=$(ls "$OUTPUT_DIR"/benchmark_*.txt 2>/dev/null || true)

    NEW_FILE=$(comm -13 <(echo "$BEFORE_FILES" | sort) <(echo "$AFTER_FILES" | sort))

    if [[ -n "$NEW_FILE" ]]; then
        NEW_NAME="${EXP_NAME}_${i}.txt"
        mv "$NEW_FILE" "$RUN_DIR/$NEW_NAME"

        echo "[SUCCESS] $NEW_NAME"
    else
        echo "[WARNING] No output file for i=$i"
    fi

    # 7. Clean ROS logs (IMPORTANT)
    rosclean purge -y >/dev/null 2>&1 || true

    echo "===== Done i=$i ====="
    echo

    sleep 5
done

echo "All runs completed for experiment: $EXP_NAME"