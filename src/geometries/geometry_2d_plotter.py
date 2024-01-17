import cv2
import numpy as np
from bistable_material_geom import BistableMaterialGeometry

geom = BistableMaterialGeometry(a=1.0, n=2, xn=12, yn=12, visualize=False)
bars = geom.bars
hinges = geom.hinges
nodes = geom.nodes

img = np.zeros((800, 800, 3), np.uint8)
for [p0, p1, l0] in bars:
    p0 = list(nodes[p0[0]][p0[1]])
    p1 = list(nodes[p1[0]][p1[1]])

    p0[0] = int(p0[0] * 100 + 100)
    p0[1] = int(p0[1] * 100 + 100)
    p1[0] = int(p1[0] * 100 + 100)
    p1[1] = int(p1[1] * 100 + 100)

    cv2.line(img, (p0[0], p0[1]), (p1[0], p1[1]), (255, 255, 255), 1)

for [p0, p1, p2, p3, th0, l0, type] in hinges:
    p0 = list(nodes[p0[0]][p0[1]])
    p1 = list(nodes[p1[0]][p1[1]])
    p2 = list(nodes[p2[0]][p2[1]])
    p3 = list(nodes[p3[0]][p3[1]])

    p0[0] = int(p0[0] * 100 + 100)
    p0[1] = int(p0[1] * 100 + 100)
    p1[0] = int(p1[0] * 100 + 100)
    p1[1] = int(p1[1] * 100 + 100)
    p2[0] = int(p2[0] * 100 + 100)
    p2[1] = int(p2[1] * 100 + 100)
    p3[0] = int(p3[0] * 100 + 100)
    p3[1] = int(p3[1] * 100 + 100)

    if type == 'facet':
        color = (0, 255, 0)
    elif type == 'fold':
        if th0 - np.pi > 0:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
    else:
        color = (255, 0, 255)
    cv2.line(img, (p0[0], p0[1]), (p1[0], p1[1]), color, 1)

img0 = img.copy()
cv2.imshow('img', cv2.flip(img0, 0))
cv2.imwrite('SimpleCutPattern.png', cv2.flip(img0, 0))
# cv2.imwrite('MiuraOriPattern.png', cv2.flip(img, 0))

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for [p0, p1, p2, p3, th0, l0, type] in hinges:
    p0 = list(nodes[p0[0]][p0[1]])
    p1 = list(nodes[p1[0]][p1[1]])
    p2 = list(nodes[p2[0]][p2[1]])
    p3 = list(nodes[p3[0]][p3[1]])

    p0[0] = int(p0[0] * 100 + 100)
    p0[1] = int(p0[1] * 100 + 100)
    p1[0] = int(p1[0] * 100 + 100)
    p1[1] = int(p1[1] * 100 + 100)
    p2[0] = int(p2[0] * 100 + 100)
    p2[1] = int(p2[1] * 100 + 100)
    p3[0] = int(p3[0] * 100 + 100)
    p3[1] = int(p3[1] * 100 + 100)

    cv2.circle(img, (p0[0], p0[1]), 5, (0, 0, 255), -1)
    cv2.circle(img, (p1[0], p1[1]), 5, (0, 255, 0), -1)
    cv2.circle(img, (p2[0], p2[1]), 5, (255, 0, 0), -1)
    cv2.circle(img, (p3[0], p3[1]), 5, (0, 255, 255), -1)

    cv2.imshow('img', cv2.flip(img, 0))
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    img = img0.copy()

# cv2.imwrite('MiuraOriPattern.png', cv2.flip(img, 0))
