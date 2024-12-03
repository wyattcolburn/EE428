import skimage as ski
from skimage import color, filters, feature
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
import numpy as np
import imageio
import matplotlib 
matplotlib.use('gtk3agg') # use this for my vim setup
from matplotlib import pyplot as plt


class MagicWand:
    def __init__(self,calibration_path,R):
        """ Loads calibration from file and stores ball radius.
            Arguments:
                calibration_path: path to calibration file
                R: ball radius in cm
        """
        self.focal, self.centerx, self.centery = np.loadtxt(calibration_path,delimiter=' ')
        self.R = R

        plt.ion()

        self.fig = plt.figure()
        self.ax_image = self.fig.add_subplot(1,3,1)
        self.ax_edges = self.fig.add_subplot(1,3,2)
        self.ax_points = self.fig.add_subplot(1,3,3,projection='3d')
        self.ax_points.set_xlabel('X')
        self.ax_points.set_ylabel('Y')
        self.ax_points.set_zlabel('Z')

    def preprocess_image(self,image):
        """ Convert to grayscale and detect edges.
            Arguments:
                image: RGB image [H,W,3]
            Returns:
                Edge map [H,W]
        """
        # WRITE CODE HERE
        gray_image = ski.color.rgb2gray(image) #this turns each pixel into a float 0-1
        edges = canny(gray_image) 
        return edges 

    def detect_circles(self, edges, image):
        """ Detect circles in edge map.
            Arguments:
                image: edge map [H,W]
            Returns:
                List of tuples (x, y, radius)
        """       
        # Define range of radii to detect
        hough_radii = np.arange(40, 55, 2)
        
        # Perform Hough Circle Transform
        hough_res = hough_circle(edges, hough_radii)
        
        # Find the most prominent circle
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
        """ 
        # Plot detected circles
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
            image[circy, circx] = (220, 20, 20)  # Highlight detected circle in red
        
        # Display the result
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image)
        ax.axis('off')  # Hide axes for clarity
        plt.show()
        """
        return list(zip(cx, cy, radii))

    def calculate_ball_position(self,x,y,r):
        """ Calculate ball's (X,Y,Z) position in world coordinates
            Arguments:
                x,y: 2D position of ball in image
                r: radius of ball in image
            Returns:
                X,Y,Z position of ball in world coordinates
        """
        # WRITE CODE HERE
        Z = (self.focal * self.R) / r
        X = ((x - self.centerx) * Z) / self.focal
        Y = ((y - self.centery) * Z) / self.focal

        return X,Y,Z

    def draw_ball(self,x,y,r,Z,image):
        """ Draw circle on ball and write depth estimate in center
            Arguments:
                x,y,r: 2D position and radius of ball
                Z: estimated depth of ball
        """
        self.ax_image.imshow(image)  # Show the image in the axes (e.g., your frame)
        circle = plt.Circle((x,y),r,fill=False)
        self.ax_image.add_patch(circle)
        self.ax_image.text(x,y,f'{int(Z)} cm')
    
    def project(self,X,Y,Z):
        """ Pinhole projection.
            Arguments:
                X,Y,Z: 3D point
            Returns:    
                (x,y) 2D location of projection in image
        """
        x = (self.focal * X / Z) + self.centerx
        y = (self.focal * Y / Z) + self.centery

        return x,y

    def draw_line_2d(self,x1,y1,x2,y2):
        """ Draw a 2D line
            Arguments:
                x1,y1: 2D position of first line endpoint
                x2,y2: 2D position of second line endpoint
        """
        self.ax_image.plot((x1,x2),(y1,y2),'b-')

    def draw_line_3d(self,X1,Y1,Z1,X2,Y2,Z2):
        """ Draw a 3D line.  (This should call draw_line_2d.)
            Arguments:
                X1,Y1,Z1: 3D position of first line endpoint
                X2,Y2,Z2: 3D position of second line endpoint
        """
        x1,y1 = self.project(X1, Y1, Z1)
        x2, y2 = self.project(X2,Y2,Z2)
        self.draw_line_2d(x1,y1,x2,y2)

    def draw_bounding_cube(self,X,Y,Z):
        """ Draw bounding cube around 3D point, with radius R
            Arguments:
                image: image on which to draw
                X,Y,Z: 3D center point of cube
        """
        R = self.R

        self.draw_line_3d( X-R, Y-R, Z-R, X+R, Y-R, Z-R )
        self.draw_line_3d( X-R, Y+R, Z-R, X+R, Y+R, Z-R )

        self.draw_line_3d( X-R, Y-R, Z-R, X-R, Y+R, Z-R )
        self.draw_line_3d( X+R, Y-R, Z-R, X+R, Y+R, Z-R )

        self.draw_line_3d( X-R, Y-R, Z+R, X+R, Y-R, Z+R )
        self.draw_line_3d( X-R, Y+R, Z+R, X+R, Y+R, Z+R )

        self.draw_line_3d( X-R, Y-R, Z+R, X-R, Y+R, Z+R )
        self.draw_line_3d( X+R, Y-R, Z+R, X+R, Y+R, Z+R )

        self.draw_line_3d( X-R, Y-R, Z-R, X-R, Y-R, Z+R )
        self.draw_line_3d( X-R, Y-R, Z-R, X-R, Y-R, Z+R )

        self.draw_line_3d( X-R, Y+R, Z-R, X-R, Y+R, Z+R )
        self.draw_line_3d( X-R, Y+R, Z-R, X-R, Y+R, Z+R )

        self.draw_line_3d( X+R, Y-R, Z-R, X+R, Y-R, Z+R )
        self.draw_line_3d( X+R, Y-R, Z-R, X+R, Y-R, Z+R )

        self.draw_line_3d( X+R, Y+R, Z-R, X+R, Y+R, Z+R )
        self.draw_line_3d( X+R, Y+R, Z-R, X+R, Y+R, Z+R )
    
    def process_frame(self,image):
        """ Detect balls in frame, estimate 3D positions, and update plots
            Arguments:
                image: RGB image to be processed [H,W,3]
        """
        edges = self.preprocess_image(image)
        self.ax_image.imshow(image)
        self.ax_image.set_ylim(image.shape[0],0)
        self.ax_image.set_xlim(0,image.shape[1])
        self.ax_edges.imshow(edges)
        self.ax_edges.set_ylim(image.shape[0],0)
        self.ax_edges.set_xlim(0,image.shape[1])
        circles = self.detect_circles(edges, image)
        for circle in circles:
            x, y, r = circle
            X,Y,Z = self.calculate_ball_position(x,y,r)
            self.draw_ball(x,y,r,Z,image)
            self.draw_bounding_cube(X,Y,Z)
            self.ax_points.plot(X,Y,Z,'r.')
        plt.pause(0.01)
        self.ax_image.cla()


if __name__ == '__main__':
    import imageio.v3 as iio
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('video',help='path to input video file')
    parser.add_argument('--output',help='path to output video file (optional)')
    parser.add_argument('--calibration',default='iphone_calib.txt',help='path to calibration file')
    parser.add_argument('--ball_radius',type=float,default=3,help='radius of ball in cm')
    args = parser.parse_args()

    wand = MagicWand(calibration_path=args.calibration,R=args.ball_radius)
    for frame in iio.imiter(args.video):
        wand.process_frame(frame)
