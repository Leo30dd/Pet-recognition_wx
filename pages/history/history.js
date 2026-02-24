// pages/history/history.js
Page({
  data: {
    historyList: []
  },

  onShow: function () {
    this.loadHistory();
  },

  loadHistory() {
    // 加上 .reverse() 让最新的在最上面显示，如果存的时候已经unshift了就不用
    const list = wx.getStorageSync('history_list') || [];
    this.setData({ historyList: list });
  },

  // 跳转到详情页
  goToDetail(e) {
    const breed = e.currentTarget.dataset.breed;
    wx.navigateTo({
      url: '/pages/detail/detail?breed=' + breed,
    })
  },

  // 删除单条记录
  deleteItem(e) {
    const idToDelete = e.currentTarget.dataset.id;
    
    wx.showModal({
      title: '提示',
      content: '确定删除这条记录吗？',
      success: (res) => {
        if (res.confirm) {
          // 1. 过滤掉要删除的那个ID
          const newList = this.data.historyList.filter(item => item.id !== idToDelete);
          
          // 2. 更新页面
          this.setData({ historyList: newList });
          
          // 3. 更新本地缓存
          wx.setStorageSync('history_list', newList);
          
          wx.showToast({ title: '已删除', icon: 'none' });
        }
      }
    })
  },

  // 清空所有
  clearHistory() {
    wx.showModal({
      title: '警告',
      content: '确定要清空所有记录吗？此操作不可恢复。',
      confirmColor: '#ff4d4f',
      success: (res) => {
        if (res.confirm) {
          wx.removeStorageSync('history_list');
          this.setData({ historyList: [] });
        }
      }
    })
  }
})