import cv2
import numpy as np
import matplotlib.pyplot as plt
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
import time

# Motor control pins
pwm_pin1 = 12
pwm_pin2 = 13
left_motor_pin1 = 18
left_motor_pin2 = 17
right_motor_pin1 = 22
right_motor_pin2 = 23

def process_frame(frame: np.ndarray, cross_section_kwargs: dict) -> np.ndarray:

    # read image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.rotate(image, cv2.ROTATE_180)

    # crop image
    image = image[image.shape[0]//2:,:]

    # std normalize image
    image = (image - np.mean(image)) / np.std(image)
    #image = image.astype(np.int8)

    # making image binary
    image = (image < -0.9).astype('uint8')

    # Define the kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))

    # Apply the opening operation
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Frame", opening * 255)


    start = cross_section_kwargs.get("start", 120)
    end = cross_section_kwargs.get("end", 200)
    line_count = cross_section_kwargs.get("line_count", 3)

    stepsize = int(np.ceil((end - start) / line_count))

    lines=opening[start:end:stepsize,:].astype('int8')
    lines = np.hstack([np.zeros((line_count, 1)), lines, np.zeros((line_count, 1))])
    res = np.abs(lines[:,0:-1] - lines[:,1:])

    indices = np.nonzero(res)
    indices = indices[1]
    indices = indices.reshape((-1, 4))

    x1=indices[:,0]
    x2=indices[:,1]
    x3=indices[:,2]
    x4=indices[:,3]
    avg1 = (x1+x2)/2
    avg2 = (x3+x4)/2
    avg_mid = (avg1+avg2)/2
    img_mid = opening.shape[1]/2
    weights = [1,2,3]
    weighted_midpoints = np.dot(weights[-avg_mid.shape[0]:],avg_mid)
    summation=np.sum(weights)

    weighted_error= weighted_midpoints/summation - img_mid
    
    slope = (avg_mid[0] - avg_mid[2]) / stepsize

    return weighted_error, slope

def init_camera():
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    raw_capture = PiRGBArray(camera, size=(640, 480))
    time.sleep(0.1)
    #
    return camera, raw_capture

def init_car():
    #TODO: Init GPIO
    # Initialize GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pwm_pin1, GPIO.OUT)
    GPIO.setup(pwm_pin2, GPIO.OUT)
    GPIO.setup(left_motor_pin1, GPIO.OUT)
    GPIO.setup(left_motor_pin2, GPIO.OUT)
    GPIO.setup(right_motor_pin1, GPIO.OUT)
    GPIO.setup(right_motor_pin2, GPIO.OUT)

    # Create PWM objects
    pwm1 = GPIO.PWM(pwm_pin1, 100)
    pwm2 = GPIO.PWM(pwm_pin2, 100)

    pwm1.start(0)
    pwm2.start(0)

    return pwm1, pwm2

def set_motor_speed(pwm, speed):
    pwm_pin1 = 12
    pwm_pin2 = 13
    left_motor_pin1 = 18
    left_motor_pin2 = 17
    right_motor_pin1 = 22
    right_motor_pin2 = 23

    if speed <= 0:
        GPIO.output(left_motor_pin1, GPIO.HIGH)
        GPIO.output(left_motor_pin2, GPIO.LOW)
        GPIO.output(right_motor_pin1, GPIO.HIGH)
        GPIO.output(right_motor_pin2, GPIO.LOW)
    else:
        GPIO.output(left_motor_pin1, GPIO.LOW)
        GPIO.output(left_motor_pin2, GPIO.HIGH)
        GPIO.output(right_motor_pin1, GPIO.LOW)
        GPIO.output(right_motor_pin2, GPIO.HIGH)
    pwm.ChangeDutyCycle(abs(speed))


def controller(error, slope, pwm1, pwm2, delta_time, Kp, Ki, Kd, prev_error, integral):
    #pid 
    prop_err = Kp * error
    integ_err = Ki * (((prop_err+prev_error)/2)*delta_time + integral)
    slope_err = Kd * (prop_err-prev_error) / delta_time
    pid = prop_err + integ_err + slope_err
    integral = integ_err
    return pid, error, integral

def signal_to_motor(pid, pwm1, pwm2):
    #pwm
    # print(pid, 13 + pid/4.5, 13 - pid/4.5)
    set_motor_speed(pwm1, 13 - pid/4.5)
    set_motor_speed(pwm2, 13 + pid/4.5)

def loop(frame: np.ndarray, pwm1, pwm2, prev_error, integral, prev_time):
    delta_time = time.time() - prev_time


def main():
    camera, raw_capture = init_camera()
    pwm1, pwm2 = init_car()
    prev_error, integral = 0, 0
    prev_time = time.time()
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        # Extract the frame as a NumPy array
        image = frame.array
        try:
            weighted_error, slope = process_frame(image, cross_section_kwargs={"start": 120, "end": 200, "line_count": 3})
            delta_time = time.time() - prev_time
            prev_time = time.time()
            # print(delta_time)
            pid, prev_error, integral = controller(weighted_error + 3 * slope, slope, pwm1, pwm2, delta_time, 0.2, 0.1, 0.03, prev_error, integral)
            signal_to_motor(pid, pwm1, pwm2)
        except:
            pass

        key = cv2.waitKey(1)


        # Exit if 'q' is pressed
        if key & 0xFF == ord('q'):
            break
        raw_capture.truncate(0)
        #cv2.imshow("Frame", image)



if __name__ == "__main__":
    main()