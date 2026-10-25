// Close sidebar when clicking outside on mobile
document.addEventListener('click', function(e) {
    const sidebar = document.getElementById('sidebar');
    const toggle = document.querySelector('.menu-toggle');
    if (sidebar && sidebar.classList.contains('open')) {
        if (!sidebar.contains(e.target) && e.target !== toggle) {
            sidebar.classList.remove('open');
        }
    }
});

// Highlight active nav items based on current URL
document.querySelectorAll('.nav-item').forEach(item => {
    if (item.href && window.location.pathname === new URL(item.href).pathname) {
        item.classList.add('active');
    }
});