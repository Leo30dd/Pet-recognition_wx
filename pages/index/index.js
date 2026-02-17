// pages/index/index.js
Page({
  data: {
    imageSrc: '',
    result: null,
    isLoading: false,
    errorMsg: ''
  },

  chooseImage: function() {
    if (this.data.isLoading) return;
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        this.setData({
          imageSrc: res.tempFiles[0].tempFilePath,
          result: null,
          errorMsg: ''
        });
      },
      fail: (err) => {
        console.log('取消选择', err);
      }
    })
  },

  startRecognize: function() {
    if (!this.data.imageSrc) {
      wx.showToast({ title: '请先选择图片', icon: 'none' });
      return;
    }

    this.setData({ isLoading: true, errorMsg: '', result: null });

    // 【注意】请务必确保这里的 IP 地址是你电脑的局域网 IP
    // 如果你在本机模拟器调试，也可以尝试用 'http://127.0.0.1:5000/predict'
    // 如果用真机调试，必须用局域网 IP (如 192.168.x.x)
    const uploadUrl = 'http://192.168.1.10:5000/predict'; 

    wx.uploadFile({
      url: uploadUrl,
      filePath: this.data.imageSrc,
      name: 'file',
      success: (res) => {
        console.log('后端原始返回:', res.data);
        try {
          const data = JSON.parse(res.data);
          
          if (res.statusCode === 200 && !data.error) {
            // 成功！
            this.setData({ result: data });
          } else {
            // 业务错误
            this.setData({ errorMsg: data.error || '识别失败，请重试' });
          }
        } catch (e) {
          this.setData({ errorMsg: '结果解析失败' });
        }
      },
      fail: (err) => {
        console.error(err);
        this.setData({ errorMsg: '连接服务器失败，请检查网络' });
      },
      complete: () => {
        this.setData({ isLoading: false });
      }
    });
  }
})