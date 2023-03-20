# Google Colab設定方式

# Define functions to connect to Google and change directories
```python
def connectDrive():
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

def changeDirectory(path):
    import os
    original_path = os.getcwd()
    os.chdir(path)
    new_path = os.getcwd()
    print("Original path: ",original_path)
    print("New path: ",new_path)
```

```python
# Connect to Google Drive
connectDrive()

# Change path
import os
if os.path.isfile("/content/drive/My Drive/github"):
    pass
else:
    !mkdir "/content/drive/My Drive/github"

changeDirectory("/content/drive/My Drive/github")
```
# Clone Git repo
```python
!git clone https://github.com/aapatel09/handson-unsupervised-learning.git
```
# Access repo
```python
%cd handson-unsupervised-learning
```
# Install libraries
```python
!pip install awscli==1.19.28
```
# Download data from AWS S3
```python
!aws s3 cp s3://handson-unsupervised-learning/datasets/ datasets --recursive --no-sign-request
```
