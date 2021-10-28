# Color Transfer between Images by Python

This git include python code based on <code>numpy,matplotlib,skimage,sklearn,argparse</code> to transfer color from an image to another image(in .jpeg format), which is a code implementation of the paper <https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf>. 

You can run the code by 

```bash
python color_transfer.py -b YOUR_COLOR -w YOUR_IMAGE -n CLUSTER_NUM
```

Then you can see the result showed by plt. 

For an example, you can take color from <code>color.jpeg</code> to <code>want.jpeg</code> by

```bash
python color_transfer.py -b color.jpeg -w want.jpeg -n 32
```

Then, you can see this image: 

![avatar](result.jpeg)