document.addEventListener("DOMContentLoaded", function () {

    const searchBox = document.getElementById("historySearch");
    const table = document.getElementById("historyTable");

    if (!searchBox || !table) return;

    searchBox.addEventListener("keyup", function () {

        const filter = searchBox.value.toLowerCase();
        const rows = table.querySelectorAll("tbody tr");

        rows.forEach(function(row) {

            const text = row.textContent.toLowerCase();

            if (text.includes(filter)) {
                row.style.display = "";
            } else {
                row.style.display = "none";
            }

        });

    });

});