<template>
    <div id="data-content">
        <div id="confusion-matrix-container">
            <confusion-matrix ref="matrix" id="confusion-matrix" @clickCell="clickConfusionCell" :showColor="true">
            </confusion-matrix>
        </div>
    </div>
</template>

<script>
import ConfusionMatrix from './ConfusionMatrix.vue';
import axios from 'axios';

export default {
    components: {ConfusionMatrix},
    name: 'DataView',
    data() {
        return {
        };
    },
    methods: {
        clickConfusionCell: function(d) {
            const store = this.$store;
            axios.post(store.getters.URL_GET_IMAGES_IN_MATRIX_CELL, {
                labels: d.rowNode.leafs,
                preds: d.colNode.leafs,
            }).then(function(response) {
                const images = response.data;
                console.log(images);
            });
        },
    },
};
</script>

<style scoped>
#data-content {
    width: 100%;
    height: 100%;
    display: flex;
}

#confusion-matrix-container {
  width: 100%;
  height: 100%;
  display: flex;
  margin: 10px 10px 10px 10px;
}
</style>
