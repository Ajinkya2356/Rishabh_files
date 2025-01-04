from pyueye import ueye
import numpy as np
import cv2
import os

desired_width = 470
desired_height = 420

save_directory = "./section 2 clear"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

predefined_sections = [(45, 9, 457, 408)]

hCam = ueye.HIDS(0)
sInfo = ueye.SENSORINFO()
cInfo = ueye.CAMINFO()
rectAOI = ueye.IS_RECT()
ueye.is_InitCamera(hCam, None)
ueye.is_GetCameraInfo(hCam, cInfo)
ueye.is_GetSensorInfo(hCam, sInfo)
ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED)
ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
camera_width = int(rectAOI.s32Width)
camera_height = int(rectAOI.s32Height)
bitspixel = 24
mem_ptr = ueye.c_mem_p()
mem_id = ueye.int()
ueye.is_AllocImageMem(hCam, camera_width, camera_height, bitspixel, mem_ptr, mem_id)
ueye.is_SetImageMem(hCam, mem_ptr, mem_id)
ueye.is_CaptureVideo(hCam, ueye.IS_WAIT)

cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Video", desired_width, desired_height)

print("Press 'c' to capture predefined sections, 'q' to quit")


def capture_predefined_sections(frame, sections):
    for i, (x_start, y_start, x_end, y_end) in enumerate(predefined_sections):
        roi_image = frame[y_start + 1 : y_end - 1, x_start + 1 : x_end - 1]
        filename = os.path.join(save_directory, f"section_image_{i}.png")
        cv2.imwrite(filename, roi_image)
        print(f"Captured and saved: {filename}")


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def align_images(master, input):
    master_preprocessed = preprocess_image(master)
    input_preprocessed = preprocess_image(input)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(master_preprocessed, None)
    keypoints2, descriptors2 = sift.detectAndCompute(input_preprocessed, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    aligned_image = cv2.warpPerspective(input, H, (master.shape[1], master.shape[0]))
    _, mask_master = cv2.threshold(master, 128, 255, cv2.THRESH_BINARY)
    mask_contours, _ = cv2.findContours(mask_master[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aligned_thresh = cv2.threshold(aligned_image, 160, 255, cv2.THRESH_BINARY_INV)[1]
    result = cv2.bitwise_or(mask_master, aligned_thresh)
    result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Master Image", mask_master)
    cv2.imshow("Aligned Image", aligned_thresh)
    cv2.imshow("Logical OR Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result, aligned_image,mask_contours


def find_defect(img_path):
    master_path = "./section 2 clear/New_Master.png"
    input_path = img_path

    master = cv2.imread(master_path)
    input = cv2.imread(input_path)

    input = cv2.resize(input, (master.shape[1], master.shape[0]))
    difference, aligned_image,mask_contours = align_images(master, input)
    cv2.imwrite("Difference.png", difference)

    # Convert the difference image to grayscale
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    _, thresholded_diff = cv2.threshold(difference_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imwrite("Thresholded Diff.png", thresholded_diff)
    contours, _ = cv2.findContours(
        thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    highlighted_image = aligned_image.copy()
    for contour in contours:
        found = True
        for c in mask_contours:
            if cv2.matchShapes(contour, c, 1, 0.0) > 200:
                found = False
                break
        if found:
            cv2.drawContours(highlighted_image, [contour], -1, (0, 0, 255), 2)

    cv2.imshow("Defects", highlighted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


while True:
    image_buffer = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
    ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
    ueye.is_CopyImageMem(hCam, mem_ptr, mem_id, image_buffer.ctypes.data)
    resized_frame = cv2.resize(image_buffer, (desired_width, desired_height))
    for x_start, y_start, x_end, y_end in predefined_sections:
        cv2.rectangle(resized_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
    cv2.imshow("Live Video", resized_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        capture_predefined_sections(resized_frame, predefined_sections)
        print("Sections captured.")

        # Load the captured section image
        section_image_path = os.path.join(save_directory, "section_image_0.png")

        find_defect(section_image_path)
    if key == ord("q"):
        break
ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
ueye.is_ExitCamera(hCam)
cv2.destroyAllWindows()