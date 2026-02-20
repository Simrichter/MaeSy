import os
import cv2

def visualize_annotations(input_dir, output_dir):
    """
        Zeichnet Bounding Boxen (im YOLO Format) in Bilder

        Args:
            :param input_dir: Ordner mit Bildern und Annotationen (.txt im YOLO Format)
            :param output_dir: Ordner, in dem die annotierten Bilder gespeichert werden
    """
    if output_dir!="" and os.path.exists(output_dir):
        raise ValueError(f"Failed: Output directory {output_dir} does not exist. Leave unspecified to create a 'visualized' subfolder in the input directory.")
    if output_dir=="":
        output_dir = os.path.join(input_dir, "visualized")
    os.makedirs(output_dir, exist_ok=True)

    # Kamera-Auflösung (anpassen, falls anders)
    img_width = 640#544
    img_height = 480#448

    color_coding = {
        0: (0, 0, 255), # Rot für "soccer ball"
        1: (255, 0, 0), # Blau für "Robot"
        2: (0, 255, 0) # Grün für "LineCrossing"
    }

    name_coding ={
        0: "Ball",
        1: "Robot",
        2: "PenaltyCross" # (Keine 27 Beschriftungen erwünscht), alternativ: LineCrossing
    }


    for file in os.listdir(input_dir):
        if file.endswith(".png"):
            img_path = os.path.join(input_dir, file)
            txt_path = os.path.join(input_dir, file.replace(".png", ".txt"))

            # Bild laden
            img = cv2.imread(img_path)

            # Annotation einlesen
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        cls, x_center, y_center, w, h = map(float, parts)

                        # YOLO -> Pixel-Koordinaten
                        cx = int(x_center * img_width)
                        cy = int(y_center * img_height)
                        bw = int(w * img_width)
                        bh = int(h * img_height)

                        x_min = cx - bw // 2
                        y_min = cy - bh // 2
                        x_max = cx + bw // 2
                        y_max = cy + bh // 2

                        # Bounding Box zeichnen
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color_coding[int(cls)], 1)
                        cv2.putText(img, f"{name_coding[int(cls)]}", (x_min+2, (y_max + y_min)//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_coding[int(cls)], 1)
                        print(f"Verarbeite {file}... Box: ({x_min},{y_min})-({x_max},{y_max})")

            # Annotiertes Bild speichern
            out_path = os.path.join(output_dir, file)
            cv2.imwrite(out_path, img)

    print(f"Fertig! Annotierte Bilder liegen in: {output_dir}")