// app.js
App({
  globalData: {
    // 本地使用http://127.0.0.1:5000
    // 真机查询ip地址
    baseUrl: 'http://127.0.0.1:5000', 
    userId: null
  },

  onLaunch() {
    // 模拟设备指纹：生成并本地存储一个唯一ID，替代登录注册流程
    let storedId = wx.getStorageSync('device_uuid');
    if (!storedId) {
      storedId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 8);
      wx.setStorageSync('device_uuid', storedId);
    }
    this.globalData.userId = storedId;
  }
})