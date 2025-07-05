# Multi-task Remote-Sensing Feature Pyramid Network (MTLRSFPN++)

## Abstract
<p align="justify">Advancements in optical remote sensing technology, supported by the ease of access to satellite data, have enabled the processing of Earth’s surface images for various applications. However, the presence of particles in the atmosphere and human activities may introduce atmospheric haze that degrades the visibility of these images. To address this issue, this research proposes RSHazeFPN++, a lightweight dehazing deep learning method for satellite images. RSHazeFPN++ is a modification of the RSHazeNet architecture, designed to reduce the impact of non-homogeneous haze. RSHazeFPN++ integrates adaptive multi-scale learning modules to select relevant features through Gated Res2Net and SKFusion. This research also adapts the proposed RSHazeFPN++ model to handle multitask dehazing and land segmentation, jointly performing both tasks simultaneously. The adaptation structure is called MTLRSFPN++. For the dehazing experiment on the benchmarked SateHaze1k dataset, RSHazeFPN++ achieves an average PSNR score of 25.264 and an average SSIM of 0.899. For the multitask experiment of dehazing and land segmentation, the MTLRSFPN++ branch achieved an average PSNR score of 26.245 and an average SSIM of 0.844 on the GID-5 data. Meanwhile, the segmentation accuracy obtains an average OA score of 0.852 and an mIoU of 0.731.</p>

<div align="center">
  <img src="https://raw.githubusercontent.com/pinantyo/MTLRSFPN/refs/heads/main/image/Proposed.jpg" alt="RSHazeFPN++"/>
</div>


## Environment

* pytorch==1.11.0
* torchvision==0.12.0 
* torchaudio==0.11.0
* numpy==1.26.4 (numpy<2)
* torchmetrics==1.5.2
* thop
* torch-summary

## Dehazing Experiments

<p align="center"><strong>Table 1. Method Complexity and Computational Estimation</strong></p>

<table align="center">
  <tr>
    <th>Method</th>
    <th>MACs (G)</th>
    <th>Params (M)</th>
    <th>Inference (S)</th>
  </tr>
  <tr>
    <td>RSHazeNet (Baseline)</td>
    <td>10.036</td>
    <td>1.190</td>
    <td>0.64336</td>
  </tr>
  <tr>
    <td><strong>RSHazeNet+</strong></td>
    <td>8.007</td>
    <td>0.827</td>
    <td>0.53169</td>
  </tr>
  <tr>
    <td><strong>RSHazeFPN+</strong</td>
    <td>10.758</td>
    <td>0.915</td>
    <td>0.78270</td>
  </tr>
  <tr>
    <td><strong>RSHazeFPN++</strong></td>
    <td>11.067</td>
    <td>0.921</td>
    <td>0.67797</td>
  </tr>
</table>

<p align="center"><strong>Table 2. SateHaze1k Dehazing Performance</strong></p>

<table align="center">
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="2">Thin</th>
    <th colspan="2">Moderate</th>
    <th colspan="2">Thick</th>
    <th colspan="2">Average</th>
  </tr>
  <tr>
    <th>PSNR</th>
    <th>SSIM</th>
    <th>PSNR</th>
    <th>SSIM</th>
    <th>PSNR</th>
    <th>SSIM</th>
    <th>PSNR</th>
    <th>SSIM</th>
  </tr>
  <tr>
    <td>RSHazeNet (Baseline)</td>
    <td>24.619</td>
    <td>0.904</td>
    <td>25.407</td>
    <td>0.928</td>
    <td>22.003</td>
    <td>0.821</td>
    <td>24.010</td>
    <td>0.884</td> 
  </tr>
  <tr>
    <td><strong>RSHazeNet+</strong></td>
    <td>25.320</td>
    <td>0.913</td>
    <td>26.759</td>
    <td>0.933</td>
    <td>22.617</td>
    <td>0.841</td>
    <td>24.899</td>
    <td>0.896</td>
  </tr>
  <tr>
    <td><strong>RSHazeFPN+</strong</td>
    <td>25.620</td>
    <td>0.915</td>
    <td>25.974</td>
    <td>0.929</td>
    <td>23.177</td>
    <td>0.846</td>
    <td>24.924</td>
    <td>0.899</td>
  </tr>
  <tr>
    <td><strong>RSHazeFPN++</strong></td>
    <td>25.739</td>
    <td>0.916</td>
    <td>27.009</td>
    <td>0.934</td>
    <td>23.045</td>
    <td>0.845</td>
    <td>25.264</td>
    <td>0.899</td>
  </tr>
</table>

## Links
* [SateHaze1k Remote-sensing Benchmark Dataset](https://www.dropbox.com/scl/fi/wtga5ltw5vby5x7trnp0p/Haze1k.zip?rlkey=70s52w3flhtif020nx250jru3&dl=0)
* [GID-5 (Gaofen Image Dataset-5 Classes)](https://x-ytong.github.io/project/GID.html)
* [Pre-trained Weights](https://drive.google.com/drive/folders/1NflRtLwh2lo-TquvQwLgWwNypbhn3xq0?usp=sharing)
