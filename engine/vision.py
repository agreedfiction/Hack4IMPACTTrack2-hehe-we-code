import cv2
import os
from ultralytics import YOLO

# Environment Override for AMD 780M
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

# Load the model once globally
MODEL = YOLO('yolo11n.pt') 

def extract_frame_metadata(inference_results):
    """Parses raw YOLO results into a structured summary."""
    frame_summary = []
    if not inference_results.boxes:
        return frame_summary

    for box in inference_results.boxes:
        label = inference_results.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        
        # Mapping categories
        PRODUCE = ["tomato", "onion", "potato", "person", "cell phone"]
        category = "produce" if label in PRODUCE else "defect"
        
        frame_summary.append({
            "label": label,
            "confidence": round(confidence, 2),
            "box": coords,
            "category": category
        })
    return frame_summary

# ONLY RUNS IF SCRIPT IS EXECUTED DIRECTLY (FOR TESTING)
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("--- Vani-Check Diagnostic Mode ---")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        results = MODEL(frame, verbose=False)[0]
        metadata = extract_frame_metadata(results)
        
        if metadata: print(metadata)
        
        cv2.imshow("Vani-Check View", results.plot())
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5): cv2.waitKey(1)

def calculate_quality_score(metadata):
    """
    Calculates a 0-100 score based on defect area ratios.
    """
    base_score = 100.0
    
    # 1. Find the primary produce area
    produce = [obj for obj in metadata if obj['category'] == 'produce']
    if not produce:
        return 0 # No produce detected
    
    # Use the largest produce item found
    main_item = max(produce, key=lambda x: (x['box'][2]-x['box'][0]) * (x['box'][3]-x['box'][1]))
    item_area = (main_item['box'][2]-main_item['box'][0]) * (main_item['box'][3]-main_item['box'][1])
    
    # 2. Subtract penalties for defects
    defects = [obj for obj in metadata if obj['category'] == 'defect']
    
    for d in defects:
        defect_area = (d['box'][2]-d['box'][0]) * (d['box'][3]-d['box'][1])
        # Calculate ratio of defect to total item
        ratio = defect_area / item_area
        
        # Apply weight (Default 1.5 for bruise, 5.0 for rot)
        weight = 5.0 if d['label'] == 'rot' else 1.5
        penalty = (ratio * weight) * 100
        base_score -= penalty

    # 3. Clip score between 0 and 100
    return max(0, min(100, round(base_score, 1)))

# Check if there are any detections
if results.boxes:
    for box in results.boxes:
        # 1. Get the class name
        class_id = int(box.cls[0])
        label = results.names[class_id]
        
        # 2. Get the confidence score
        confidence = float(box.conf[0])
        
        # 3. Get the coordinates (xmin, ymin, xmax, ymax)
        coords = box.xyxy[0].tolist()
        
        print(f"Detected: {label} | Confidence: {confidence:.2f} | Box: {coords}")
else:
    print("No objects detected in this frame.")
