const app = getApp();

Page({
  data: {
    historyList: []
  },

  onShow: function () {
    this.fetchHistory();
  },

  // 1. 获取列表
  fetchHistory() {
    wx.request({
      url: `${app.globalData.baseUrl}/history`,
      method: 'GET',
      data: { user_id: app.globalData.userId },
      success: (res) => {
        if (res.data.code === 200) {
          this.setData({ historyList: res.data.data });
        }
      }
    });
  },

  // 2. 跳转到百科详情
  goToDetail(e) {
    const breed = e.currentTarget.dataset.breed;
    if (breed === "未识别为狗") {
      wx.showToast({ title: '无法查看详情', icon: 'none' });
      return;
    }
    wx.navigateTo({
      url: '/pages/detail/detail?breed=' + breed,
    })
  },

  // 3. 删除单条记录
  deleteItem(e) {
    const id = e.currentTarget.dataset.id;
    wx.showModal({
      title: '提示',
      content: '确定删除这条记录吗？',
      success: (res) => {
        if (res.confirm) {
          wx.request({
            url: `${app.globalData.baseUrl}/delete_history`,
            method: 'POST',
            data: { id: id }, // 发送要删除的 ID 给后端
            success: (resp) => {
              if (resp.data.code === 200) {
                wx.showToast({ title: '已删除' });
                this.fetchHistory(); // 刷新列表
              }
            }
          })
        }
      }
    })
  },

  // 4. 清空所有
  clearHistory() {
    wx.showModal({
      title: '警告',
      content: '确定要清空吗？',
      success: (res) => {
        if (res.confirm) {
          wx.request({
            url: `${app.globalData.baseUrl}/history/clear`,
            method: 'POST',
            data: { user_id: app.globalData.userId },
            success: (resp) => {
              if (resp.data.code === 200) {
                this.setData({ historyList: [] });
              }
            }
          })
        }
      }
    })
  }
})