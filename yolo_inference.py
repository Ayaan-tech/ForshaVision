from ultralytics import YOLO
model = YOLO('training/models/best.pt')
result= model.predict('input_videos/input_1.mp4', save=True, save_dir='output_videos')
print(result[0])

print("==========Log==========")
for box in result[0].boxes:
    print(box)
