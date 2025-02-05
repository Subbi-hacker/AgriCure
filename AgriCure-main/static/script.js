document.addEventListener('DOMContentLoaded', function () {
    var tabLinks = document.querySelectorAll('.tab-link');
    var tabContents = document.querySelectorAll('.tab-content');

    tabLinks.forEach(function(link) {
        link.addEventListener('click', function(event) {
            event.preventDefault();
            var targetTab = this.getAttribute('data-tab');

            tabContents.forEach(function(content) {
                content.classList.remove('active');
            });
            tabLinks.forEach(function(tabLink) {
                tabLink.classList.remove('active');
            });

            document.getElementById(targetTab).classList.add('active');
            this.classList.add('active');

            var offsetTop = document.getElementById(targetTab).offsetTop;
            window.scrollTo({ top: offsetTop, behavior: 'smooth' });
        });
    });
});
function toggleNav() {
    var navBar = document.getElementById('navBar');
    navBar.classList.toggle('closed'); 
}

