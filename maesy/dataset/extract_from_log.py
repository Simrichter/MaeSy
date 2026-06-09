import os
from pathlib import Path

import numpy as np
import cv2
import yaml
from sensor_msgs.msg import Image
from tqdm import tqdm


def extract_mcap(
bag_path,
topic_name,
output_dir,
exact_match
):
    """
    Extract images from an MCAP log file and save them to a specified directory.
    Args:
        :param bag_path: Path to the MCAP log file.
        :param topic_name: List of topic names to extract images from.
        :param output_dir: Directory to save the extracted images.
        :param exact_match: Whether to match topic names exactly or just by the last part (e.g. "/image_left_raw" matches "/camera/image_left_raw")
    """
    # Attempt ROS2-related imports
    try:
        # Catch Unresolved Reference Errors, as this only works if Ros2 is sourced in executing terminal
        from rclpy.serialization import deserialize_message
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    except ModuleNotFoundError:
        print("_" * 60)
        raise EnvironmentError("ROS2 Python libraries not found. If you need rosbag log functionality, source your ROS2 workspace")

    save_path = Path(output_dir)/Path(bag_path).name.removesuffix(".mcap")
    topic_dirs = {topic: save_path/(Path(topic.lstrip("/"))) for topic in topic_name}
    for topic_dir in topic_dirs.values():
        os.makedirs(topic_dir, exist_ok=True)
    pbar = None
    meta_path = Path(bag_path).parent /"metadata.yaml"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = yaml.safe_load(f)
        total_messages = metadata["rosbag2_bagfile_information"]["message_count"]
        pbar = tqdm(total=total_messages)

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id='mcap'),
        ConverterOptions('', '')
    )

    print("="*60)
    print("Extrating images from log:")
    # print("="*60)

    counter = 0
    while reader.has_next():
        topic, data, t = reader.read_next()
        if not exact_match:
            topic = f'/{topic.split("/")[-1]}'
        if topic in topic_dirs:
            msg = deserialize_message(data, Image)

            # Direkt aus Buffer lesen
            img = np.frombuffer(msg.data, dtype=np.uint8)

            # Shape bestimmen
            if msg.encoding == "rgb8":
                img = img.reshape((msg.height, msg.width, 3))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            elif msg.encoding == "bgr8":
                img = img.reshape((msg.height, msg.width, 3))

            elif msg.encoding == "mono8":
                img = img.reshape((msg.height, msg.width))
            elif msg.encoding.lower() == "nv12":
                img = np.frombuffer(msg.data, dtype=np.uint8)
                img = img.reshape((msg.height * 3 // 2, msg.width))
                img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)
            else:
                print(f"Unsupported encoding: {msg.encoding}")
                continue

            filename = os.path.join(topic_dirs[topic], f'image_{counter:06d}.png')
            cv2.imwrite(filename, img)
            counter += 1
        if pbar is not None:
            pbar.update(1)
    pbar.close()
    print("Done.")
    print("="*60)

