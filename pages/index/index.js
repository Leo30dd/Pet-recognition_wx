// pages/index/index.js
const app = getApp();

Page({
  data: {
    imageSrc: '',
    result: null,
    isLoading: false,
    errorMsg: ''
  },

  // 选择图片
  chooseImage: function() {
    if (this.data.isLoading) return;
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        this.setData({
          imageSrc: res.tempFiles[0].tempFilePath,
          result: null, // 换图后清空旧结果
          errorMsg: ''
        });
      },
      fail: (err) => {
        console.log('取消选择', err);
      }
    })
  },

  // 核心识别功能 (上传到后端，后端负责存数据库)
  startRecognize: function() {
    if (!this.data.imageSrc) {
      wx.showToast({ title: '请先选择图片', icon: 'none' });
      return;
    }

    this.setData({ isLoading: true, errorMsg: '', result: null });

    // 从 app.js 获取全局配置的后端地址
    const baseUrl = app.globalData.baseUrl; 
    const uploadUrl = `${baseUrl}/predict`;
    const userId = app.globalData.userId;

    wx.uploadFile({
      url: uploadUrl,
      filePath: this.data.imageSrc,
      name: 'file',
      // 【关键】带上 user_id，告诉后端是谁在传图
      formData: {
        'user_id': userId
      },
      success: (res) => {
        console.log('后端返回:', res.data);
        try {
          const response = JSON.parse(res.data);
          
          if (res.statusCode === 200 && response.code === 200) {
            // 成功！后端已存数据库，前端只需展示结果
            this.setData({ result: response.data });
          } else {
            // 业务错误
            const msg = response.msg || response.error || '识别失败';
            this.setData({ errorMsg: msg });
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
  
  // 跳转到详情页
  goToDetail() {
    if (this.data.result && this.data.result.breed) {
      const breed = this.data.result.breed;
      if (breed === "未识别为狗") {
        wx.showToast({ title: '无法查看详情', icon: 'none' });
        return;
      }
      wx.navigateTo({
        url: '/pages/detail/detail?breed=' + breed,
      })
    }
  },

  // 跳转到关于页面
  goToAbout() {
    wx.navigateTo({ url: '/pages/about/about' })
  },

  // 跳转到历史页面 
  goToHistory() {
    wx.navigateTo({ url: '/pages/history/history' })
  },

  // --- 分享功能 (保留逻辑，虽然没认证不能用) ---
  onShareAppMessage: function () {
    let title = '智能狗狗品种识别';
    if (this.data.result) {
      title = `我发现了一只 ${this.data.result.breed}，置信度 ${this.data.result.confidence}%！`;
    }
    return {
      title: title,
      path: '/pages/index/index',
      imageUrl: this.data.imageSrc || ''
    }
  },

  onShareTimeline: function () {
    let title = '智能狗狗品种识别';
    if (this.data.result) {
      title = `AI识别结果：${this.data.result.breed}`;
    }
    return {
      title: title,
      imageUrl: this.data.imageSrc || ''
    }
  }
})