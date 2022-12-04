# Walking on spheres

This repo is an educative implementation for walking spheres used to render diffusion curve images.

It should be plug and play when cuda is installed and compiled with visual studio.

## Controls

The application has several debug/educational features which can be controlled using the keyboard

### Screens

The application consits of several screens which can be navigated using the left and right arrow screen.

The first screen shows a image which encodes the distance to the closest diffusion curve from its coordinate

The second image shows the color value of the closest point on the closest diffusion curve.

And the last screen show the current solution

### Keys

The space bar is used as pause/play to gather as many samples as possible in the render loop and will rapidly refine the solution to the exact solution.

The R key resets the solution back to zero.

The D key will take exactly one sample if we are not taking continuous samples.

The C key will simulate taking a sample at the location of the cursor and show the path it could take to a boundary condition.
