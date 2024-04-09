from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import base64
from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2 as cv
import numpy as np
import time
import os

# Azure Computer Vision credentials
subscription_key = "7df694aa70d749e69992f2c4f6ca7eeb"
endpoint = "https://docscanner620.cognitiveservices.azure.com/"

# Authenticate
computervision_client = ComputerVisionClient(
    endpoint, CognitiveServicesCredentials(subscription_key))


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
#cors = CORS(app, origins='*')
CORS(app)

@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Health check route
@app.route('/')
def hello() :
    return 'Health check.'

######### Boundary detection ######

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def decode_base64_image(base64_string):
    # Remove the prefix 'data:image/jpeg;base64,' from the base64 string
    base64_string = base64_string.split(',')[1]
    # Decode the base64 string into bytes
    image_bytes = base64.b64decode(base64_string)
    # Convert the bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the image array into an OpenCV image
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return image

def detect_document_boundary(image):
    if image is None:
        print("Input image is None")
        return None

    original = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blurred, 75, 200)
    contours, _ = cv.findContours(
        edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contours found")
        return original

    docCnt = None
    height, width = image.shape[:2]
    image_area = height * width
    largest_contour_area = cv.contourArea(contours[0])
    if largest_contour_area > 0.8 * image_area:
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        for c in contours:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    if docCnt is None:
        print("No document contour found")
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    warped = four_point_transform(original, docCnt.reshape(4, 2))
    return warped


@app.route("/api/boundary_detection", methods=['POST'])
def boundary_detection():
    # Get the base64-encoded image string from the form data
    image_base64 = request.form.get('image')
    if image_base64 is None:
        return 'No image provided', 400

    image = decode_base64_image(image_base64)

    processed_image = detect_document_boundary(image)
    
    _, img_encoded = cv.imencode('.jpg', processed_image)
    cv.destroyAllWindows()
    return img_encoded.tobytes(), 200, {'Content-Type': 'image/jpeg'}

###################################

##### Character Recognition #######

def deskew_image(img):
    """
    Attempt to deskew an image based on text orientation
    """
    coords = np.column_stack(np.where(img > 0))
    angle = cv.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (w, h),
                            flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated


def enhance_image(img):
    """
    Enhance the image quality for better OCR results
    """
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding after Gaussian filtering
    binarized = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv.THRESH_BINARY, 11, 2)

    # Deskew the image
    deskewed = deskew_image(binarized)

    return deskewed

def preprocess_for_ocr(img, enhance=1):
    """
    Apply a series of preprocessing steps to prepare an image for OCR.
    """
    # Step 1: Convert to grayscale to reduce the effect of color variations
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Step 2: Contrast adjustment to make the text stand out
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0     # Brightness control (0-100)
    contrast_enhanced = cv.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Step 3: Noise reduction to reduce the effect of image artifacts
    noise_reduced = cv.fastNlMeansDenoising(contrast_enhanced, None, 30, 7, 21)

    # Step 4: Binarization to create a clear distinction between text and background
    _, binarized = cv.threshold(
        noise_reduced, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Step 5: Deskewing the image to correct text alignment
    deskewed = deskew_image(binarized)

    # Step 6: Scale the image to a standard resolution to maintain consistency across different image sizes
    final_img = cv.resize(deskewed, None, fx=enhance,
                          fy=enhance, interpolation=cv.INTER_LINEAR)

    # Step 7: Median blur to smooth out the image while preserving the edges
    final_img = cv.medianBlur(final_img, 3)

    return final_img

# Recognize text from an image using Azure OCR
def recognize_text(img):
    """
    Recognize text from an image using Azure OCR
    :param img: input image
    :return: recognized text
    """
    # Preprocess the image for OCR
    processed_img = preprocess_for_ocr(img)

    # Writing the processed image to disk for Azure OCR consumption
    temp_filename = 'temp.jpg'
    cv.imwrite(temp_filename, processed_img)

    with open(temp_filename, "rb") as image_stream:
        read_response = computervision_client.read_in_stream(
            image_stream, raw=True)

    # Cleaning up the temporary file as soon as it's no longer needed
    os.remove(temp_filename)

    # Extracting the operation ID for the OCR process
    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]

    # Polling for the OCR result
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower() not in ['notstarted', 'running']:
            break
        print('Waiting for result...')
        time.sleep(1)

    # Collecting text from the OCR results
    if read_result.status == OperationStatusCodes.succeeded:
        text = [
            line.text for text_result in read_result.analyze_result.read_results for line in text_result.lines]
        recognized_text = "\n".join(text)
    else:
        recognized_text = ""

    return recognized_text

@app.route("/api/character_recognition", methods=['POST'])
def character_recognition():
    image_base64 = request.form.get('image')
    if image_base64 is None:
        return 'No image provided', 400

    # Decode base64 image
    image = decode_base64_image(image_base64)

    # Recognize text using Azure Cognitive Services Computer Vision OCR
    recognized_text = recognize_text(image)

    # Encode processed image to return
    _, img_encoded = cv.imencode('.jpg', image)
    cv.destroyAllWindows()

    # Convert image to base64 string
    img_bytes = img_encoded.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Return processed image and the recognized text
    return jsonify({'image': img_base64, 'recognized_text': recognized_text}), 200


###################################

####### Color Correction ##########

def correct_color(image, brightness=1.2):
    # Convert the image to LAB color space
    LABimg = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    # Calculate average values of 'a' and 'b' channels
    avg_1 = np.average(LABimg[:, :, 1])
    avg_2 = np.average(LABimg[:, :, 2])

    # Apply color correction by adjusting the 'a' and 'b' channels based on the luminance channel 'L'
    LABimg[:, :, 1] = LABimg[:, :, 1] - ((avg_1 - 128) * (LABimg[:, :, 0] / 255.0) * brightness)
    LABimg[:, :, 2] = LABimg[:, :, 2] - ((avg_2 - 128) * (LABimg[:, :, 0] / 255.0) * brightness)

    # Convert the image back to BGR color space
    balanced_image = cv.cvtColor(LABimg, cv.COLOR_LAB2BGR)

    return balanced_image

@app.route("/api/color_correction", methods=['POST'])
def color_correction():
    # Get base64-encoded image string
    image_base64 = request.form.get('image')
    if image_base64 is None:
        return 'No image provided', 400

    image = decode_base64_image(image_base64)

    corrected_image = correct_color(image)

    # Encode corrected image to return
    _, img_encoded = cv.imencode('.jpg', corrected_image)
    cv.destroyAllWindows()
    return img_encoded.tobytes(), 200, {'Content-Type': 'image/jpeg'}

###################################

#### Remove Background Noise ######

def remove_noise(image):
    # convert to grascale, provide gaussian blur
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(img, (0,0), sigmaX=33, sigmaY=33)

    # divide
    divide = cv.divide(img, blur, scale=255)  

    # setting otsu threshold
    threshold = cv.threshold(divide, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    # applying morphology
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,2))
    morph = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel)


    return morph

@app.route("/api/remove_noise", methods=['POST'])
def remove_noise_route():
    # Get base64-encoded image string
    image_base64 = request.form.get('image')
    if image_base64 is None:
        return 'No image provided', 400

    image = decode_base64_image(image_base64)

    processed_image = remove_noise(image)

    # Encode processed image to return
    _, img_encoded = cv.imencode('.jpg', processed_image)
    cv.destroyAllWindows()
    return img_encoded.tobytes(), 200, {'Content-Type': 'image/jpeg'}


###################################

# Api test route
@app.route("/api/users", methods=['GET'])
def users():
    return jsonify (
        {
            "users": [
                "Member1", 
                "Member2"
            ]
        }
    )

if __name__ == "__main__":
    app.run(debug=True, port=8080)