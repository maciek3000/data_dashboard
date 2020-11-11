// Listeners for Features Menu
var elems = document.querySelectorAll(".features-menu .single-feature");
var i;

for (i=0; i < elems.length; i++) {
    var func = (function(elem, elements) {
        return function() {
            var elem_text = elem.innerText.split(". ")[1];
            // Bokeh included here to make sure that it is loaded by the time the event should fire
            var dropdown = Bokeh.documents[0].get_model_by_name("features_dropdown");
            dropdown.value = elem_text;

            // Changing the style for active feature
            var active = "active-feature";
            var i;
            for (i = 0; i < elements.length; i++) {
                elements[i].classList.remove(active);
            }
            elem.classList.add(active);

            // Changing Title for the page
            var feature_title = document.querySelector(".chosen-feature");
            feature_title.innerText = elem_text;

        };
    });
    elems[i].addEventListener("click", func(elems[i], elems));
};

// Burger Menu Button and Menu X close button
var burger_menu_button = document.querySelector(".burger-button");
var x_button = document.querySelector(".close-button");
for (button of Array(burger_menu_button, x_button)) {
    button.addEventListener("click", function() {
        var features_menu = document.querySelector(".features-menu");
        features_menu.classList.toggle("active-menu");
})};

window.onload = document.querySelectorAll(".features-menu .single-feature")[0].classList.add("active-feature");