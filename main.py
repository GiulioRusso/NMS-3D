import pandas as pd
import torch
from pandas import read_csv

from nms_3d import *


def main():

    print("| ---------------------------------- |\n"
          "| Non-Maximum Suppression 3D example |\n"
          "| ---------------------------------- |\n")

    # ----------- #
    # READ COORDS #
    # ----------- #
    # read bounding box coordinates from a .csv file
    prediction_boxes_df = read_csv(filepath_or_buffer='bbox-coords-before-nms-3d.csv')
    iou_threshold = 0.5

    # convert DataFrame to PyTorch tensors
    prediction_boxes = torch.tensor(data=prediction_boxes_df.values,
                                    dtype=torch.float32)

    # --- #
    # NMS #
    # --- #
    # perform 3D non-maximum suppression
    best_boxes = nms_3d(prediction_boxes=prediction_boxes,
                        iou_threshold=iou_threshold,
                        debug=True)

    # convert the tensor back to DataFrame after NMS
    best_boxes_df = pd.DataFrame(best_boxes.numpy(),
                                 columns=['SCORE', 'X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX'])

    # save the result into a CSV file
    best_boxes_df.to_csv(path_or_buf='bbox-coords-after-nms-3d.csv',
                         index=False)

    # ---- #
    # DRAW #
    # ---- #
    # call the function to draw prediction_boxes_df
    plot_3d_boxes(boxes_df=prediction_boxes_df,
                  title='Prediction Boxes Before NMS',
                  save_html=True,
                  html_filename_path='prediction_boxes_before_nms.html')

    # call the function to draw best_boxes_df
    plot_3d_boxes(boxes_df=best_boxes_df,
                  title='Best Boxes After NMS',
                  save_html=True,
                  html_filename_path='best_boxes_after_nms.html')


if __name__ == "__main__":
    main()