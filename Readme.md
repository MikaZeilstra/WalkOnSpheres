# Walking on spheres

This repo is an educative implementation for walking spheres used to render diffusion curve images.

It should be plug and play when CUDA 11.8 is installed and compiled with Visual Studio 2022.

## Controls

The application has several debug/educational features which can be controlled using the keyboard.

### Screens

Several can be navigated using the left and right arrow screen.

The first screen shows only the curves for the diffusion curves,

The second screen starts of blank and is the screen which contains the current solution

The third screen shows an image which encodes the distance to the closest diffusion curve from its coordinate

The fourth image shows the color value of the closest point on the closest diffusion curve.

And the last screen show the Laplacian of the current solution since we try to solve for the Laplacian to be zero this can be interpreted as the error

### Keys

The space bar is used as pause/play to gather as many samples as possible in the render loop and will rapidly refine the solution to the exact solution.

The R key resets the solution back to zero.

The D key will take exactly one sample if we are not taking continuous samples.

The C key will simulate taking a sample at the location of the cursor and show the path it could take to a boundary condition.

## Runtime stats

The application also measures several statistics to display to the user

The window name contains the amount of samples currently taken per pixel and the average time taken per pixel in ms.

Lastly, the console shows the amount of time it took to allocate and upload the curve control points to the GPU, the time preprocessing took and the time taken for the last sample.