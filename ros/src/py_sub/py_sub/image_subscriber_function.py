# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from torchvision.models.resnet import resnet50
import torchvision.transforms as T
from glob import glob
import PIL

from .mtcnn import MTCNN


class MinimalImageSubscriber(Node):
    def __init__(self):
        super().__init__("deploy_BP_minimal_image_subscriber")
        self.subscription = self.create_subscription(
            Image, "web_cam", self.listener_callback, 10
        )
        self.subscription  # prevent unused variable warning
        self.cvbr = CvBridge()
        self.i = 0
        self.save_path = None
        self.model = resnet50(num_classes=8631)
        weights = torch.load(
            "/home/tutorial_ws/src/py_sub/resource/resnet50.pth", map_location="cpu"
        )
        self.model.load_state_dict(weights, strict=True)
        self.model.fc = torch.nn.Identity()
        self.model.cuda()
        self.model.eval()

        self.transformations = T.Compose(
            [np.float32, T.ToTensor(), T.Resize((160, 160))]
        )

        self.mtcnn = MTCNN(
            image_size=160,
            margin=14,
            selection_method="center_weighted_size",
            device="cuda",
        )

        self.gallery_paths = glob("/home/tutorial_ws/src/py_sub/resource/gallery/*")
        self.gallery_names = [os.path.basename(name) for name in self.gallery_paths]

        self.gallery_features = []
        for i, path in enumerate(self.gallery_paths):
            image = cv2.imread(path)[...,::-1]
            feature = self.extract_faces(image).squeeze()
            self.gallery_features.append(feature)
        self.gallery_features = torch.stack(self.gallery_features, dim=0)

        # If a data path is provided, save the frames
        if len(sys.argv) >= 2:
            self.get_logger().info("There are %d args" % len(sys.argv))
            self.save_path = sys.argv[1]
            # check if the folder exists. If not, create one
            if not os.path.exists(self.save_path):
                self.get_logger().info("Create the folder for saving the video frames")
                os.makedirs(self.save_path)

    def listener_callback(self, msg):
        self.get_logger().info(
            "deploy BP: I received %d Image messages with frame id %s with width %d and height %d."
            % (self.i, msg.header.frame_id, msg.width, msg.height)
        )
        sender_timestamp_msg = msg.header.stamp
        print(
            "The message is received at: %s"
            % datetime.utcfromtimestamp(sender_timestamp_msg.sec).isoformat()
        )

        sender_timestamp = rclpy.time.Time.from_msg(sender_timestamp_msg)
        receiver_timestamp = self.get_clock().now()
        delay = receiver_timestamp - sender_timestamp  # obtain Duration
        print(
            "The delay at the receiver in seconds is %.9f s."
            % (delay.nanoseconds / 1e9)
        )

        # extract the image frame from the Image msg
        frame = self.cvbr.imgmsg_to_cv2(msg)
        embeddings = self.extract_faces(frame[...,::-1], display=True)

        if embeddings is not None:
            for i, embedding in enumerate(embeddings):
                distances = (self.gallery_features - embedding).norm(dim=1)
                min_dist, min_idx = distances.topk(1, largest=False)
                min_idx = min_idx.to(int)
                print(distances)
                if min_dist <= 2.5:
                    top_1 = self.gallery_names[min_idx]
                    cv2.putText(frame, top_1.split('.')[0], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if self.save_path is not None:
            file_name = os.path.join(self.save_path, "frame%06d.png" % self.i)

        # ----- Test Processing ------
        cv2.imshow("Face Detection", frame)
        cv2.waitKey(1)
    
    def extract_faces(self, frame, display=False):
        bboxes, _ = self.mtcnn.detect(frame)

        faces = []
        if bboxes is not None:
            for (x, y, w, h) in bboxes:
                x = int(x)
                y = int(y)
                x2 = int(w)
                y2 = int(h)
                try:
                    face = frame[x:x2, y:y2]
                    feature = _fixed_image_standardisation(self.transformations(face))
                    faces.append(feature)
                    if display:
                        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                except:
                    pass
        
        if len(faces) > 0:
            faces = torch.stack(faces, dim=0).cuda()
            embeddings = self.model(faces).detach().cpu()
            return embeddings
        
        return None


def _fixed_image_standardisation(image):
    processed_tensor = (image - 127.5) / 128.0
    return processed_tensor


def main():
    rclpy.init()

    minimal_image_subscriber = MinimalImageSubscriber()

    rclpy.spin(minimal_image_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
