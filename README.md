# Dynamic Rupture for Ground Motion
The repository hosts Python utilties, information, and links to datasets for simulated broadband ground motions from earthquake dynamic ruptures towards better characterizing seismic hazard for engineering applications.

## Simulated datasets
Available datasets can be downloaded through the following links <br/>
[eqdyna.0001.A.100m](https://doi.org/10.6084/m9.figshare.25561833.v1) <br/>
[eqdyna.0001.B.100m](https://doi.org/10.6084/m9.figshare.25561935.v1) <br/>

## Compute GM metrics from datasets
Leveraging on gmpe-smtk, dr4gm provides utilities to process raw simulated ground motion data for various GM metrics <br/>
such as PGV, PGA, RSA(T), CAV, etc.

dr4gm is available through docker 
```
docker pull dunyuliu/dr4gm
```
After a dataset is downloaded and unzipped, users need to nagivate to the path for datasets, and run
```
gmProcessor xMin xMax yMin yMax gridSize # to obtain 2-D maps of GM metrics given the ranges and resolution.
```
or 
```
gmProcessor x y # to obtain GM metrics for a single station.
```
