import cv2
import numpy as np
import tkinter as tk

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')

#for resizing the image
new_width = 350
new_height = 250

#for drawing the lines
left_top_x = 0
left_bottom_x = 0
right_top_x = 450
right_bottom_x = 450

while True:
        ret, frame = cam.read()

        if ret is False:
            break

        #1.the output is the original image with the original width and height
        #cv2.imshow('Original', frame)

        #2.resizing the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        cv2.imshow('Resized', resized_frame)

        #3.converting to grayscale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale', gray_frame)
        #daca s-ar face manual cream un frame gol cu np.zeros(h,w), prin doua for-uri am parcurge fiecare pixel (h, w)
        #calculam valoarea medie(nivelul general de lumina) a pixelilor colori(np.mean), apoi setam valoarea pixelilor in noul frame in tonuri de gri

        #4.selecting the road
        height, width = gray_frame.shape

        upper_right = (int(width * 0.55), int(height * 0.76))
        upper_left = (int(width * 0.45), int(height * 0.76))
        lower_left = (0, height)
        lower_right = (width, height)

        trapezoid = np.zeros((height, width), dtype=np.uint8)
        points = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)
        cv2.fillConvexPoly(trapezoid, points, color=255) #desenez trapezoidul(frame_in_which_to_draw, points_of_a_polygon, color_to_draw_with)
        #255 clearly visible

        result_frame = gray_frame * trapezoid
        cv2.imshow('Road', result_frame*255)

        #5.get a top-down view! (sometimes called a birds-eye view)
        trapezoid_bounds = np.float32([upper_right, upper_left, lower_left, lower_right]) #colturile pe care dorim sa le intindem
        screen_bounds = np.float32([np.array([new_width, 0]), np.array([0, 0]), np.array([0, new_height]),  np.array([new_width, new_height])]) #colturile unde dorim sa le intindem

        matrix = cv2.getPerspectiveTransform(trapezoid_bounds, screen_bounds) #(bounds_of_current_area, bounds_of_area_you_want_to_stretch_to)
        stretched_frame = cv2.warpPerspective(gray_frame, matrix, (new_width, new_height)) #returneaza imaginea intinsa
        cv2.imshow('Top-Down', stretched_frame)

        #6.add a bit of blur
        n=3 #daca folosim n mai mare imaginea va fi mai blurata si neaparat numar impar!
        blured_frame = cv2.blur(stretched_frame, ksize=(n, n))
        cv2.imshow('Blur', blured_frame)

        #7.do edge detection

        #sobel- pentru fiecare pixel se inmulteste fiecare element cu nr corespunzator din matrice si se aduna rezultatele intre ele
        sobel_vertical = np.float32([[-1, -2, -1],
                                     [0, 0, 0],
                                     [+1, +2, +1]])

        sobel_horizontal = np.transpose(sobel_vertical)

        stretched_frame_float32 = np.float32(stretched_frame)
        stretched_frame_float32_copy1 = stretched_frame_float32.copy()
        stretched_frame_float32_copy2 = stretched_frame_float32.copy()

        vertical_edges = cv2.filter2D(stretched_frame_float32_copy1, -1, sobel_vertical)
        horizontal_edges = cv2.filter2D(stretched_frame_float32_copy2, -1, sobel_horizontal)

        #urmatoarele 4 linii sunt pentru a afisa cele 2 imagini separate(una verticala, cealalta orizontala)
        #vertical_edges_display = cv2.convertScaleAbs(vertical_edges)
        #horizontal_edges_display = cv2.convertScaleAbs(horizontal_edges)
        #cv2.imshow('vert', vertical_edges_display)
        #cv2.imshow('hor', horizontal_edges_display)

        combined_edges = np.sqrt(np.square(vertical_edges) + np.square(horizontal_edges))
        combined_edges_display = cv2.convertScaleAbs(combined_edges)
        cv2.imshow('Sobel', combined_edges_display)

        #8.binarize the frame
        threshold_value = 127
        _, thresholded_image = cv2.threshold(combined_edges_display, threshold_value, 255, cv2.THRESH_BINARY)
        # _ este folosita ca sa indice ca nu avem nevoie de prima valoare returnata ci doar de thresholded_image. Se foloseste o valoare fixa pe care am definit-o, deci nu ne intereseaza valorile determinate automat
        cv2.imshow('Thresholded Image', thresholded_image)

        #9.get the coordinates of street markings on each side of the road
        #se utilizeaza metoda de regresie liniara pentru a gasi ecuatia liniilor care reprezinta marginile benzilor de circulatie pe fiecare parte a drumului
        thresholded_image_copy = thresholded_image.copy()
        #seteaza prima si ultima coloana a copiei la 0 (culoare neagra) pentru a elimina pixelii din margini
        thresholded_image_copy[:, :int(width * 0.1)] = 0
        thresholded_image_copy[:, -int(width * 0.1):] = 0

        split_half = width // 2 # separa imaginea in 2 parti(stanga si dreapta)
        left_half = thresholded_image_copy[:, :split_half]
        right_half = thresholded_image_copy[:, split_half:]

        #argwhere returneaza un vector ce contine coordonatele elementelor cautate, sub forma (y,x)
        #np.argwhere pentru a gasi coordonatele pixelilor cu valori mai mari de 1 pe ambele jumatati
        argwhere_left = np.argwhere(left_half > 1)
        argwhere_right = np.argwhere(right_half > 1)
        # np.argwhere((right_half % 2)

        left_xs = argwhere_left[:, 1]
        left_ys = argwhere_left[:, 0]
        right_xs = argwhere_right[:, 1] + split_half #se aduna split_half ca sa se ajusteze coordonatele pentru că partea dreaptă este situată în a doua jumătate a imaginii
        right_ys = argwhere_right[:, 0]

        #10.find the lines that detect the edges of the lane
        #np.polynomial.polynomial.polyfit(x_list, y_list, deg = 1) folosit pt a obtine linia, polinom de grad 1
        #returneaza b si a din ecuatia y=ax+b in ordinea asta

        left_line = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
        #right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

        if len(right_xs) > 0 and len(right_ys) > 0:  # Verifică dacă coordonatele sunt nenule
                right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)
        else:
                right_line = left_line  # cazul în care coordonatele sunt nule

        left_top_y = 0
        left_bottom_y = height
        right_top_y = 0
        right_bottom_y = height

        #coordonatele x pentru punctele superioare ale benzilor de circulație stânga și dreapta
        left_top_xx = int((left_top_y - left_line[0]) / left_line[1])
        left_bottom_xx = int((left_bottom_y - left_line[0]) / left_line[1])
        #coordonatele x pentru punctele inferioare ale benzilor de circulație stânga și dreapta
        right_top_xx = int((right_top_y - right_line[0]) / right_line[1])
        right_bottom_xx = int((right_bottom_y - right_line[0]) / right_line[1])

        if 0 <= left_top_xx < split_half:
                left_top_x = left_top_xx

        if 0 <= left_bottom_xx < split_half:
                left_bottom_x = left_bottom_xx

        if split_half <= right_bottom_xx < width:
                right_bottom_x = right_bottom_xx

        if split_half < right_top_xx < width:
                right_top_x = right_top_xx

        cv2.line(thresholded_image_copy, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), (200, 0, 0), 5)
        cv2.line(thresholded_image_copy, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), (100, 0, 0), 5)
        cv2.imshow('Lines', thresholded_image_copy)

        #11.create a final visualization

        #final_left_lane = np.zeros((new_width, new_height), dtype=np.uint8)
        #cv2.line(final_left_lane, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), (255, 0, 0), 5)
        #cv2.line(imaginea_unde_se_deseneaza, coordonate_start_final, culoare, grosime, tip_linie)

        frame_left = np.zeros((new_height, new_width), dtype=np.uint8)
        cv2.line(frame_left, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), (255, 0, 0), 5)
        magicalMatrix_left = cv2.getPerspectiveTransform(screen_bounds, trapezoid_bounds)
        frame_left1 = cv2.warpPerspective(frame_left, magicalMatrix_left, (new_width, new_height))

        frame_right = np.zeros((new_height, new_width), dtype=np.uint8)
        cv2.line(frame_right, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), (255, 0, 0), 5)
        magicalMatrix_right = cv2.getPerspectiveTransform(screen_bounds, trapezoid_bounds)
        frame_right1 = cv2.warpPerspective(frame_right, magicalMatrix_right, (new_width, new_height))

        #indicii pixelilor din aceste matrice care au valori mai mari decât 1 (adică pixeli care fac parte din benzile de circulație)
        argwhere_left = np.argwhere(frame_left1 > 1)
        argwhere_right = np.argwhere(frame_right1 > 1)
        # print(right_ys,right_xs)

        right_half = frame_right1[:, split_half:] #separam matricea frame_right1 în right_half pentru a izola partea dreaptă a benzii de circulație

        left_xs = argwhere_left[:, 1]
        left_ys = argwhere_left[:, 0]
        right_xs = argwhere_right[:, 1]
        right_ys = argwhere_right[:, 0]

        frame_final = resized_frame.copy()
        frame_final[left_ys, left_xs] = (200, 50, 200)
        frame_final[right_ys, right_xs] = (50, 50, 250)

        cv2.imshow("Final", frame_final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()

