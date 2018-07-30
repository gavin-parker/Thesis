# Deep Learning for Illumination Estimation from Stereo Images
![](https://github.com/gavin-parker/Thesis/blob/master/title_img.PNG "Ground Truth vs Predicted")
In this paper we present a system for estimating the lighting in photographs and video streams of real scenes, that improves on previous work by eliminating the need for known geometry. We present a Convolutional Neural Network that exploits stereo views of an object to learn the environment lighting via reflectance. This environment lighting can then be used to realistically render and composite new objects.
Augmented Reality(AR) is a growing topic in computer graphics and computer vision, that involves superimposing virtual objects on images. The aim is to give the illusion that the object is part of the real scene. However, current AR solutions do little to estimate the lighting in the scene and so the added objects lack realism. The motion picture industry solves this problem by taking images of a mirror ball, which has known geometry and material properties. This is a slow process that must be performed beforehand, and is clearly inappropriate for live AR applications. The ability to estimate the lighting from arbitrary objects would make it possible to realistically augment existing or even live footage.
The appearance of an object is a combination of material, geometry and lighting. This makes estimation of any one of these factors from a single image a difficult task. Previous attempts at this task have relied on significant constraints; the material of the object must be very reflective, and the exact shape must be known beforehand. Furthermore current methods use a Reflectance Mapping step which comes with a significant performance penalty, and is vulnerable to inaccurate geometry estimates. We have produced a CNN that exploits learned stereo matching to infer surface properties and produce a usable lighting approximation. To do so we performed the following work:
* Replicated the work of Stamatios Georgoulis et al. in Tensorflow by building CNNs that can interpolate sparse reflectance maps and predict environment map lighting from single objects with provided geometry. Achieved equivalent results on known surface geometry and confirmed the limitations of estimated geometry.
* Implemented a dataset generator that could render objects with realistic lighting and material properties for training our model. Produced a high-detail stereo image dataset of 55,000 image-lighting pairs.
* Augmented our large dataset of HDRI environment maps with Google Street View images, automatically tone-mapped by HDR-ExpandNet.
* Created a new Siamese architecture to encode geometry from stereo images and estimate lighting conditions. This achieved similar performance to previous work without the need for known surface geometry.
* Improved upon the Siamese architecture with a Cosine Similarity step, to achieve good performance with over 2X faster inference time.

Our network is able to achieve equivalent results to previous work that relies on known geometry, and outperforms those that use estimated surface normals in both accuracy and inference speed.

Read the full thesis with results [here] (https://github.com/gavin-parker/Thesis/blob/master/dissertation.pdf "Thesis")
