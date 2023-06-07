import cv2


class draw:
    def __init__(self, image):
        self.image = image

    def put_multiple_box(self, labels, pos):
        object_num = len(labels)
        for i in range(object_num):
            cv2.rectangle(self.image, (pos[i][0], pos[i][1]), (pos[i][2], pos[i][3]), (0, 0, 255), thickness=1,
                          lineType=cv2.LINE_4,
                          shift=0)
        return self.image

    def put_single_box(self, pos, flag):
        if flag is 1:
            cv2.rectangle(self.image, (pos[0], pos[1]), (pos[2], pos[3]), (153, 153, 0), thickness=2, lineType=cv2.LINE_4,
                          shift=0)
        elif flag is 2:
            cv2.rectangle(self.image, (pos[0], pos[1]), (pos[2], pos[3]), (153, 153, 153), thickness=2, lineType=cv2.LINE_4,
                          shift=0)
        return self.image


# if __name__ == '__main__':
#     sample = cv2.imread('disasterB.jpg')
#     while True:
#         print(sample.shape[:3])
#         drawing = draw(sample)
#         position1 = [0, 0, 150, 150]
#         result = drawing.put_single_box(position1)
#         cv2.imshow("result", result)
#         cv2.waitKey(1)
#         position2 = [200, 200, 350, 350]
#         result = drawing.put_single_box(position2)
#         cv2.imshow("result", result)
#         cv2.waitKey(1)
