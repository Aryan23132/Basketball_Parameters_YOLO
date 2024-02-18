# Basketball_Parameters_YOLO
Utilizing computer vision techniques to extract meaningful parameters from the video of a man
performing basketball dribble.

----------
*Prerequisites*
-------------
- `Python 3.10` 
- Install necessary packages using `pip install -r requirements.txt`
----------
- Run the Following command:
```bash
python stealth.py
```
View `output/outputvid_ball.mp4`

----------
*Explaination*
-------------
### For Ball detection
- Used YOLOv8 model from ultralytics which is trained on COCO Dataset.
### For Counting Number of Dribbles
- After detecting the ball, the code tracks its position. If the ball's y-coordinate is greater than the previous frame's y-coordinate, it checks if the vertical distance exceeds a threshold, indicating a downward motion. If true, it increases the dribble count. This ensures only downward ball movements are counted as dribbles.
### For Velocity of Ball
- The velocity of ball is calculated after determining the vertical distance the ball moves between consecutive frames and dividing it by the time elapsed.
- This time is computed using the frame skip and the video's frame rate.
- The distance is the difference in the y-coordinate of the ball's center.
### For Dribble Frequency
- The frequency of the ball is calculated by measuring the time it takes for the ball to complete one dribble. Then the frequency is calculated by using `1/(time elapsed)` formula.

 
