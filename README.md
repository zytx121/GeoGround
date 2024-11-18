<div align="center">
<h1>GeoGround<img src="images/geoground/logo.png" height="40">: A Unified Large Vision-Language Model for Remote Sensing Visual Grounding</h1>

<div>
    <a href='https://zytx121.github.io/' target='_blank'>Yue Zhou</a><sup>1</sup>&emsp;   
    <a href='https://mc-lan.github.io/' target='_blank'>Mengcheng Lan</a><sup>1</sup>&emsp;
    <a href='https://xiangli.ac.cn/' target='_blank'>Xiang Li</a><sup>2</sup>&emsp;
    <a href='https://keyiping.wixsite.com/index' target='_blank'>Yiping Ke</a><sup>3</sup>&emsp;
    <a href='https://ee.sjtu.edu.cn/FacultyDetail.aspx?id=53&infoid=66' target='_blank'>Jiang Xue</a><sup>4</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=PnNAAasAAAAJ&hl=en' target='_blank'>Litong Feng</a><sup>5*</sup>&emsp;
    <a href='https://www.statfe.com/' target='_blank'>Wayne Zhang</a><sup>5</sup>&emsp;
</div>
<div>
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp; 
    <sup>2</sup>CS, University of Reading&emsp; 
    <sup>3</sup>CCDS, Nanyang Technological University&emsp; 
    <sup>4</sup>SEIEE, Shanghai Jiaotong University&emsp; 
    <sup>5</sup>SenseTime Research&emsp;
</div>

[![Demo](https://img.shields.io/badge/Online-Demo-red)]()
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)]()
[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/)


</div>

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

---

## 📢 Latest Updates

- 🌟 We will release the Text4Seg demo, code and datasets as soon as possible. 🌟

---

## <img src="images/geoground/logo.png" height="30"> Abstract

*Remote sensing (RS) visual grounding aims to use natural language expression to locate specific objects (in the form of the bounding box or segmentation mask) in RS images, enhancing human interaction with intelligent RS interpretation systems. Early research in this area was primarily based on horizontal bounding boxes (HBBs), but as more diverse RS datasets have become available, tasks involving oriented bounding boxes (OBBs) and segmentation masks have emerged. In practical applications, different targets require different grounding types: HBB can localize an object's position, OBB provides its orientation, and mask depicts its shape. However, existing specialized methods are typically tailored to a single type of RS visual grounding task and are hard to generalize across tasks. In contrast, large vision-language models (VLMs) exhibit powerful multi-task learning capabilities but struggle to handle dense prediction tasks like segmentation. This paper proposes GeoGround, a novel framework that unifies support for HBB, OBB, and mask RS visual grounding tasks, allowing flexible output selection. Rather than customizing the architecture of VLM, our work aims to elegantly support pixel-level visual grounding output through the Text-Mask technique. We define prompt-assisted and geometry-guided learning to enhance consistency across different signals. To support model training, we present refGeo, a large-scale RS visual instruction-following dataset containing 161k image-text pairs. Experimental results show that GeoGround demonstrates strong performance across four RS visual grounding tasks, matching or surpassing the performance of specialized methods on multiple benchmarks.*

<div align="center">
  <img src="images/geoground/framework.jpg" width=100%>
  <div style="display: inline-block; color: #999; padding: 2px;">
      GeoGround unifies box-level and pixel-level visual grounding tasks in remote sensing.
  </div>
</div>

---

## 🏆 Contributions

- **Framework.** We propose GeoGround, a novel VLM framework that unifies box-level and pixel-level RS visual grounding tasks while maintaining its inherent dialogue and image understanding capabilities.

- **Dataset.** We introduce refGeo, the largest RS visual grounding instruction-following dataset, consisting of 161k image-text pairs and 80k RS images, including a new 3D-aware aerial vehicle visual grounding dataset.

- **Benchmark.** We conduct extensive experiments on various RS visual grounding tasks, providing valuable insights for future RS VLM research and opening new avenues for research in RS visual grounding.

---

## 💬 Text-Mask & Hybrid Supervision

We propose the Text-Mask paradigm, which distills and compresses the information embedded in the mask into a compact text sequence that can be efficiently learned by VLMs. Additionally, we introduce hybrid supervision, which incorporates prompt-assisted learning (PAL) and geometry-guided learning (GGL) to fine-tune the model using three types of signals, ensuring output consistency and enhancing the model’s understanding of the relationships between different grounding types.
<p align="center">
  <img src="images/geoground/hybrid_supervision.jpg" width=70%>
</p>

---

## 🔍 refGeo Dataset

We introduce refGeo, a large-scale RS visual grounding instruction-following dataset. It consolidates four existing visual grounding datasets from RS and introduces a new aerial vehicle visual grounding dataset (AVVG). AVVG extends traditional 2D visual grounding to a 3D context, enabling VLMs to perceive 3D space from 2D aerial imagery. For each referred object, we provide HBB, OBB, and mask, with the latter automatically generated by the SAM.

<p align="center">
  <img src="images/geoground/refgeo.jpg" width=50%>
</p>

---

## 🚀 Qualitative and Quantitative results

### 📷 Referring Expression Comprehension (REC) (HBB)
GeoGround achieves the best performance across all REC benchmarks, surpassing the specialized model on the DIOR-RSVG test set. Benefiting from the wide range of image resolutions and GSD in refGeo, the fine-tuned model showed significant performance improvements on datasets with a high proportion of small objects, such as RSVG and AVVG.

<div align="center">
  <img src="images/tables/table_1.png" width=100% alt="Table_1">
</div>


<div align="center">
  <img src="images/qualitative_results/rec_hbb.png" width=100%>
  <div style="display: inline-block; color: #999; padding: 2px;">
      Performance on RSVG benchmark.
  </div>
</div>

---

### 📷 Referring Expression Comprehension (REC) (OBB)
The results demonstrate GeoGround's dominance in RS visual grounding tasks based on OBB, further validating the effectiveness of our hybrid supervision approach.

<div align="center">
  <img src="images/tables/table_2.png" width=50% alt="Table_2">
</div>


<div align="center">
  <img src="images/qualitative_results/rec_obb.png" width=100%>
  <div style="display: inline-block; color: #999; padding: 2px;">
      Performance on GeoChat benchmark.
  </div>
</div>

---

### 📷 Referring Expression Segmentation (RES)
Unlike the other VLMs, GeoGround does not require the introduction of an additional mask decoder, as it inherently possesses segmentation capabilities. Moreover, we attempt to use SAM to refine the coarse masks generated by GeoGround, which allowed GeoGround to achieve results that match the performance of the best RS referring segmentation model.


<div align="center">
  <img src="images/tables/table_3.png" width=50% alt="Table_3">
</div>

<div align="center">
  <img src="images/qualitative_results/res_1.png" width=100%>
  <div style="display: inline-block; color: #999; padding: 2px;">
      Performance on RRSIS-D benchmark.
  </div>
</div>

<div align="center">
  <img src="images/qualitative_results/res_2.png" width=100%>
  <div style="display: inline-block; color: #999; padding: 2px;">
      Performance on RRSIS-D benchmark.
  </div>
</div>

---

### 📷 Generalized Referring Expression Comprehension (GRES) (Multiple Targets)
We present an RS Generalized REC benchmark based on AVVG, which differs from standard REC in that one referring expression may correspond to multiple objects.

<div align="center">
  <img src="images/tables/table_4.png" width=50% alt="Table_4">
</div>

---

 ### 📷 Image Captioning & Visual Question Answering (VQA)
Our approach enhances object-level understanding without compromising the holistic image comprehension capabilities of VLMs.

<div align="center">
  <img src="images/tables/table_5.png" width=50% alt="Table_5">
</div>



## 📜 Citation
```bibtex
@misc{lan2024text4seg,
      title={GeoGround: A Unified Large Vision-Language Model for Remote Sensing Visual Grounding}, 
      author={Yue Zhou and Mengcheng Lan and Xiang Li and Yiping Ke and Xue Jiang and Litong Feng and Wayne Zhang},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/}, 
}
```

---
## 🙏 Acknowledgement

