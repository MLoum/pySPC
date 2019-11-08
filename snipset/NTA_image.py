import numpy as np
from microscPSF import gLXYZParticleScan

import cv2

if __name__ == "__main__":

    pixel_size_DSLR_micron = 4.1
    objective_magnification = 40.0
    pixel_size_micron = pixel_size_DSLR_micron / objective_magnification
    nb_of_pixel_X = 5472
    nb_of_pixel_Y = 3648
    dimX_micron = pixel_size_micron * nb_of_pixel_X
    dimY_micron = pixel_size_micron * nb_of_pixel_Y
    dimZ_micron = 15

    image_NTA = np.zeros((nb_of_pixel_X, nb_of_pixel_Y))

    m_params = {"M" : objective_magnification, # magnification
                "NA" : 0.95,              # numerical aperture
                "ng0" : 1.515,           # coverslip RI design value
                "ng" : 1.515,            # coverslip RI experimental value
                "ni0" : 1,           # immersion medium RI design value
                "ni" : 1,            # immersion medium RI experimental value
                "ns" : 1.33,             # specimen refractive index (RI)
                "ti0" : 150,             # microns, working distance (immersion medium thickness) design value
                "tg" : 170,              # microns, coverslip thickness experimental value
                "tg0" : 170,             # microns, coverslip thickness design value
                "zd0" : 180.0 * 1.0e+3}  # microscope tube length (in microns).


    np_particle_type = np.dtype([('x', np.float), ('y', np.float), ('z', np.float)])
    nb_particles = 150
    particles = np.zeros(nb_particles, dtype=np_particle_type)

    r = np.random.rand(nb_particles, 3)
    particles[:]['x'] = r[:, 0]*dimX_micron
    particles[:]['y'] = r[:, 1]*dimY_micron
    particles[:]['z'] = r[:, 2]*dimZ_micron

    sub_img_size_pixel = 300

    for p in particles:
        print("x : %f, y : %f, z : %f" % (p['x'], p['y'], p['z']))
        print("x_pix : %f, y_pix : %f, z_pix : %f" % (p['x']/pixel_size_micron, p['y']/pixel_size_micron, p['z']/pixel_size_micron))
        img = gLXYZParticleScan(m_params, dxy=pixel_size_micron, xy_size=sub_img_size_pixel, pz=p['z'], normalize=False)
        img = np.squeeze(img)


        xmin = int(p['x']/pixel_size_micron - sub_img_size_pixel/2)
        xmax = int(p['x']/pixel_size_micron + sub_img_size_pixel/2)
        ymin = int(p['y']/pixel_size_micron - sub_img_size_pixel/2)
        ymax = int(p['y']/pixel_size_micron + sub_img_size_pixel/2)

        if xmin < 0:
            delta = int(-xmin) + 1
            xmin = 0
            img = img[delta:, :]
        elif xmax >= nb_of_pixel_X:
            delta = (nb_of_pixel_X-1) - xmax
            xmax = nb_of_pixel_X - 1
            img = img[:delta, :]

        if ymin < 0:
            delta = int(-ymin) + 1
            ymin = 0
            img = img[:, delta:]
        elif ymax >= nb_of_pixel_Y:
            delta = (nb_of_pixel_Y-1) - ymax
            ymax = nb_of_pixel_Y - 1
            img = img[:, :delta]

        image_NTA[xmin:xmax,ymin:ymax] += img
        # print(img)
        # print(np.shape(img))

    print("image_NTA.max()", image_NTA.max())
    image_NTA /= image_NTA.max()
    image_NTA *= 65535

    # print(image_NTA)
    cv2.imshow('image_NTA', image_NTA)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('image_NTA.png', image_NTA)

