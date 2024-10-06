import torch
from pandas import read_csv
import pandas as pd
from nms_3d import *

def main():

    print("| ---------------------------------- |\n"
          "| Non-Maximum Suppression 3D example |\n"
          "| ---------------------------------- |\n")

    # ----------- #
    # READ COORDS #
    # ----------- #
    # read bounding box coordinates from a .csv file
    prediction_boxes_df = read_csv(filepath_or_buffer='./bbox-coords/bbox-coords-before-nms-3d.csv')
    iou_threshold = 0.25

    # convert the DataFrame to PyTorch tensor
    prediction_boxes = torch.tensor(prediction_boxes_df.values, dtype=torch.float32)

    # --- #
    # NMS #
    # --- #
    # perform 3D non-maximum suppression
    best_boxes = nms_3d(prediction_boxes=prediction_boxes,
                        iou_threshold=iou_threshold,
                        debug=True)

    # convert the tensor back to DataFrame after NMS
    best_boxes_df = pd.DataFrame(best_boxes,
                                 columns=['SCORE', 'X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX'])

    # save the result into a .csv file
    best_boxes_df.to_csv(path_or_buf='./bbox-coords/bbox-coords-after-nms-3d.csv', index=False)

    # ---- #
    # DRAW #
    # ---- #
    # call the function to draw the prediction boxes (before NMS)
    plot_3d_boxes(boxes=prediction_boxes,
                  title='Prediction Boxes Before NMS',
                  save_html=True,
                  html_filename_path='./output/prediction_boxes_before_nms.html',
                  color=(255, 0, 0, 0.5),
                  show_scores=True)

    # call the function to draw the best boxes (after NMS)
    plot_3d_boxes(boxes=best_boxes,
                  title='Best Boxes After NMS',
                  save_html=True,
                  html_filename_path='./output/best_boxes_after_nms.html',
                  color=(255, 0, 0, 0.5),
                  show_scores=True)


if __name__ == "__main__":
    main()
