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

// 跳转到关于页面
goToAbout() {
  wx.navigateTo({
    url: '/pages/about/about',
  })
},

// 跳转到历史页面 (如果你还没加)
goToHistory() {
  wx.navigateTo({
    url: '/pages/history/history',
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
    const uploadUrl = 'http://127.0.0.1:5000/predict'; 

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
            this.saveToHistory(data); //保存到历史记录
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
  },
  // 保存历史记录的辅助函数
  saveToHistory(resultData) {
    // 1. 获取当前时间
    const now = new Date();
    const timeStr = `${now.getMonth() + 1}月${now.getDate()}日 ${now.getHours()}:${now.getMinutes()}`;

    // 2. 构建一条记录对象
    const newRecord = {
      imageSrc: this.data.imageSrc, // 图片路径
      breed: resultData.breed,      // 品种
      confidence: resultData.confidence, // 置信度
      time: timeStr,                // 时间
      id: Date.now()                // 唯一ID
    };

    // 3. 获取旧的记录列表 (如果没有就返回空数组)
    let historyList = wx.getStorageSync('history_list') || [];

    // 4. 把新记录加到最前面
    historyList.unshift(newRecord);

    // 5. 限制记录数量 (比如只存最近 20 条)，防止缓存爆满
    if (historyList.length > 20) {
      historyList = historyList.slice(0, 20);
    }

    // 6. 存回本地
    wx.setStorageSync('history_list', historyList);
  },
  
  goToDetail() {
    if (this.data.result && this.data.result.breed) {
      wx.navigateTo({
        url: '/pages/detail/detail?breed=' + this.data.result.breed,
      })
    }
  },

  // 1. 发送给朋友
  onShareAppMessage: function () {
    let title = '智能宠物品种识别';
    let path = '/pages/index/index';
    
    // 如果当前有识别结果，分享标题就更有趣
    if (this.data.result) {
      title = `我发现了一只 ${this.data.result.breed}，置信度 ${this.data.result.confidence}%！`;
    }

    return {
      title: title,
      path: path,
      imageUrl: this.data.imageSrc || '' // 使用当前上传的图片作为分享封面
    }
  },

  // 2. 分享到朋友圈
  onShareTimeline: function () {
    let title = '智能宠物品种识别';
    if (this.data.result) {
      title = `AI识别结果：${this.data.result.breed}`;
    }
    return {
      title: title,
      imageUrl: this.data.imageSrc || ''
    }
  }
})