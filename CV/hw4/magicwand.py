from matplotlib import pyplot as plt
import skimage
import numpy as np

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
        pass
    
    def detect_circles(self,edges):
        """ Detect circles in edge map.
            Arguments:
                image: edge map [H,W]
            Returns:
                List of tuples (x, y, radius)
        """
        # WRITE CODE HERE
        pass

    def calculate_ball_position(self,x,y,r):
        """ Calculate ball's (X,Y,Z) position in world coordinates
            Arguments:
                x,y: 2D position of ball in image
                r: radius of ball in image
            Returns:
                X,Y,Z position of ball in world coordinates
        """
        # WRITE CODE HERE

    def draw_ball(self,x,y,r,Z):
        """ Draw circle on ball and write depth estimate in center
            Arguments:
                x,y,r: 2D position and radius of ball
                Z: estimated depth of ball
        """
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
        # WRITE CODE HERE
        pass

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
        # WRITE CODE HERE
        pass

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
        circles = self.detect_circles(edges)
        for circle in circles:
            x, y, r = circle
            X,Y,Z = self.calculate_ball_position(x,y,r)
            self.draw_ball(x,y,r,Z)
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