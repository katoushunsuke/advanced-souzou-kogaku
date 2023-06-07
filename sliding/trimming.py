import cv2


class trimming:
    def __init__(self, image) -> object:
        self.original = cv2.imread(image)
        self.width, self.height, self.channels = self.original.shape[:3]
        print(self.width, self.height)

    def split(self, num):
        cell_height = int(self.height / num)
        cell_width = int(self.width / num)

        cropped = []
        for i in range(num):
            for j in range(num):
                cropped.append(
                    self.original[cell_width * i:cell_width * (i + 1), cell_height * j:cell_height * (j + 1)])

        return cropped

    def export_pos(self, num):
        cell_height = int(self.height / num)
        cell_width = int(self.width / num)

        position = []
        for i in range(num):
            for j in range(num):
                # position.append([cell_width * i, cell_width * (i + 1), cell_height * j, cell_height * (j + 1)])
                # position.append([cell_width * i, cell_height * j, cell_width * (i + 1), cell_height * (j + 1)])
                position.append([cell_height * j, cell_width * i, cell_height * (j + 1), cell_width * (i + 1)])

        return position

    def select(self, pos, label):
        top = int(pos[1])
        bottom = int(pos[3])
        left = int(pos[0])
        right = int(pos[2])

        selected = self.original[top:bottom, left:right]

        # cv2.putText(selected, text=label, org=(5, selected.shape[0] - 5), fontFace=0, fontScale=0.2,color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        return selected


"""
my_img = 'disaster2.jpeg'
# my_img = cv2.imread(my_img)

cut = trimming(my_img)
for i in range(4*4):
    cv2.imshow('im', cut.split(4)[i])
    cv2.waitKey(2000)
"""
