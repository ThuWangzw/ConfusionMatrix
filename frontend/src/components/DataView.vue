<template>
    <div id="data-content">
        <div id="confusion-matrix-container">
            <confusion-matrix v-if="matrixType==='confusion'" ref="matrix" id="confusion-matrix" :showColor="true" :confusionMatrix="confusionMatrix">
            </confusion-matrix>
            <numerical-matrix v-else-if="matrixType==='numerical'" ref="numerical"
                :numericalMatrix="numericalMatrix" :numericalMatrixType="numericalMatrixType"></numerical-matrix>
        </div>
    </div>
</template>

<script>
import ConfusionMatrix from './ConfusionMatrix.vue';
import NumericalMatrix from './NumericalMatrix.vue';
import axios from 'axios';

export default {
    components: {ConfusionMatrix, NumericalMatrix},
    name: 'DataView',
    data() {
        return {
            matrixType: 'numerical',
            numericalMatrixType: 'size',
            confusionMatrix: undefined,
            numericalMatrix: {
                'partitions': ['N', 0, 0.3, 0.6, 1],
                'matrix': [[0, 0, 0, 0], [0, 10, 2, 3], [0, 0, 8, 1], [0, 1, 0, 9]],
            },
        };
    },
    methods: {
    },
    mounted: function() {
        const store = this.$store;
        const that = this;
        axios.post(store.getters.URL_GET_CONFUSION_MATRIX)
            .then(function(response) {
                that.confusionMatrix = response.data;
            });
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
